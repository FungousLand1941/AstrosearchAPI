
import httpx
import pytest

from models import CatalogDefinition, CatalogQueryError, ResponseParseError, validate_target
from providers import TapProvider

XML = '''<?xml version="1.0"?><VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE><TABLE>
<FIELD name="Name" datatype="char" arraysize="*"/>
<FIELD name="RAJ2000" datatype="double" unit="deg"/>
<FIELD name="DEJ2000" datatype="double" unit="deg"/>
<FIELD name="Obs" datatype="char" arraysize="*"/>
<DATA><TABLEDATA><TR><TD>real-format-record</TD><TD>187.2779</TD><TD>2.0524</TD><TD>2018-06-01T00:00:00</TD></TR></TABLEDATA></DATA>
</TABLE></RESOURCE></VOTABLE>'''

def catalog(**params):
    return CatalogDefinition(name='test', provider='tap', wavelength='radio', table='"J/ApJ/914/42/table5"',
        endpoint='https://vizier.cds.unistra.fr/viz-bin/votable', epoch='Obs', epoch_format='jd',
        parameters={'protocol':'vizier_asu', 'columns':['Name','RAJ2000','DEJ2000','Obs'],
                    'ra_field':'RAJ2000','dec_field':'DEJ2000','id_field':'Name',**params})

@pytest.mark.asyncio
async def test_asu_coordinates_epoch_and_real_endpoint_provenance():
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=XML, headers={'content-type':'application/xml'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result=await TapProvider(client).query(catalog(),validate_target(187.2779,2.0524),30)
    assert len(result)==1
    assert result[0].epoch == pytest.approx(2018.413415468857, abs=1e-9)
    assert result[0].provenance['endpoint'].endswith('/viz-bin/votable')
    assert result[0].data['Obs']=='2018-06-01T00:00:00'
    assert result.meta['protocol']=='vizier_asu'
    assert requests[0].url.params['-c']=='187.277900000+2.052400000'
    assert requests[0].url.params['-sort']=='_r'
    assert requests[0].url.params['-out.max']=='201'

@pytest.mark.asyncio
@pytest.mark.parametrize('params',[{'where':'Flux > 2'},{'columns':['RAJ2000 * 15 AS ra']}])
async def test_asu_never_discards_unsupported_constraints(params):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:pytest.fail('must not query'))) as client:
        with pytest.raises(CatalogQueryError):
            await TapProvider(client).query(catalog(**params),validate_target(187,2),30)

@pytest.mark.asyncio
async def test_asu_missing_fields_are_not_silently_empty():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,text=XML))) as client:
        with pytest.raises(ResponseParseError,match='omitted requested columns'):
            await TapProvider(client).query(catalog(columns=['Name','RAJ2000','DEJ2000','Flux']),validate_target(187,2),30)

@pytest.mark.asyncio
async def test_asu_sexagesimal_positions_use_cds_decimal_coordinates():
    xml=XML.replace('<FIELD name="RAJ2000" datatype="double" unit="deg"/>',
        '<FIELD name="RAJ2000" datatype="char" arraysize="*"/>').replace(
        '<FIELD name="DEJ2000" datatype="double" unit="deg"/>',
        '<FIELD name="DEJ2000" datatype="char" arraysize="*"/>').replace(
        '<DATA>', '<FIELD name="_RAJ2000" datatype="double" unit="deg"/><FIELD name="_DEJ2000" datatype="double" unit="deg"/><DATA>').replace(
        '<TD>187.2779</TD><TD>2.0524</TD>', '<TD>12 29 06.696</TD><TD>+02 03 08.64</TD>').replace(
        '</TR>', '<TD>187.2779</TD><TD>2.0524</TD></TR>')
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,text=xml))) as client:
        result=await TapProvider(client).query(catalog(),validate_target(187.2779,2.0524),30)
    assert len(result)==1 and result[0].ra==pytest.approx(187.2779)
    assert result[0].data['RAJ2000']=='12 29 06.696'

@pytest.mark.asyncio
async def test_d25_fallback_keeps_large_distant_galaxies_and_truncation():
    from types import SimpleNamespace

    from alerts import HYPERLEDA_DEFINITION, AlertEnricher
    from models import catalog_from_dict
    from providers import QueryResult
    # A small nearby galaxy, a large farther galaxy whose ellipse reaches the
    # alert, a small far galaxy, and a non-galaxy inside the cone.
    def row(ra, size, kind='G'):
        return SimpleNamespace(ra=ra,dec=0,data={'logD25':size,'OType':kind})
    source_rows=[row(.005,-1),row(1,3),row(1,0),row(.001,4,'*')]
    class Provider:
        async def query(self, cat, target, radius):
            assert cat.max_rows==20000 and radius==21600
            assert cat.parameters['protocol']=='vizier_asu'
            assert 'where' not in cat.parameters
            return QueryResult(source_rows,{'truncated':True})
    enricher=object.__new__(AlertEnricher)
    enricher.host_radius_arcsec=60
    enricher.host_service=SimpleNamespace(providers={'tap':Provider()})
    result=await enricher._d25_asu(SimpleNamespace(ra=0,dec=0),catalog_from_dict('hyperleda_d25',HYPERLEDA_DEFINITION))
    assert list(result)==source_rows[:2]
    assert result.meta['truncated'] is True
    assert result.meta['asu_full_cone_rows']==4
