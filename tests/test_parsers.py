"""Parser unit tests against real recorded archive payloads and synthetic edge cases."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fixture_io import FIXTURES

from models import (
    CatalogQueryError,
    ResponseParseError,
    parse_csv_records,
    parse_csv_table,
    parse_ipac_records,
    parse_ipac_table,
    parse_json_records,
    parse_json_table,
    parse_votable_records,
    parse_votable_table,
    votable_query_status,
)


def body(target: str, catalog: str, idx: int = 0) -> bytes:
    return (FIXTURES / target / f"{catalog}.{idx}.body").read_bytes()


# ---------------------------------------------------------------------------
# JSON shapes
# ---------------------------------------------------------------------------


def test_tap_json_metadata_data_rows_are_zipped_by_column_name() -> None:
    table = parse_json_table(body("3c273", "gaia_dr3"))
    assert table.format == "json-tap"
    assert len(table.rows) == 1
    row = table.rows[0]
    assert row["source_id"] == 3700386905605055360  # 64-bit int kept exact (> 2**53)
    assert row["source_id"] > 2**53
    assert abs(row["ra"] - 187.2779158387) < 1e-8
    col = table.column("ra_error")
    assert col is not None and col.unit == "mas" and col.ucd == "stat.error;pos.eq.ra"
    assert table.column("ref_epoch").unit == "yr"


def test_cds_json_without_unit_keys_parses() -> None:
    table = parse_json_table(body("3c273", "simbad"))
    assert table.rows[0]["main_id"] == "3C 273"
    # SIMBAD omits 'unit' for unitless columns; computed alias has only name/datatype.
    assert table.column("match_dist").ucd is None
    assert table.column("coo_err_maj").unit == "mas"


def test_mast_info_data_rows_are_zipped() -> None:
    table = parse_json_table(body("3c273", "panstarrs_dr2"))
    assert len(table.rows) == 3
    names = {c.name for c in table.columns}
    assert {"objID", "raMean", "decMean", "raMeanErr", "distance"} <= names
    assert table.rows[0]["objID"] in ("110461872779253351", 110461872779253351)


def test_exoplanet_archive_bare_record_list() -> None:
    table = parse_json_table(body("hd209458", "exoplanet_archive"))
    assert table.format == "json-records"
    assert table.rows[0]["pl_name"] == "HD 209458 b"
    assert table.columns == []


def test_skyserver_sqlsearch_rows_are_unwrapped() -> None:
    rows = parse_json_records(body("3c273", "sdss"))
    assert len(rows) == 1
    assert rows[0]["objID"] == 1237651735760142397


def test_xamin_request_key_is_parsed() -> None:
    rows = parse_json_records((FIXTURES / "errors" / "xamin_nvss_position.0.body").read_bytes())
    assert rows and rows[0]["name"] == "NVSS J122906+020305"


def test_array_rows_without_metadata_raise_instead_of_being_dropped() -> None:
    with pytest.raises(ResponseParseError):
        parse_json_records({"data": [[1.0, 2.0], [3.0, 4.0]]})
    with pytest.raises(ResponseParseError):
        parse_json_records([[187.0, 2.0]])


def test_row_length_mismatch_raises() -> None:
    payload = {"metadata": [{"name": "ra"}, {"name": "dec"}], "data": [[1.0, 2.0, 3.0]]}
    with pytest.raises(ResponseParseError, match="3 values"):
        parse_json_table(payload)


def test_metadata_without_data_raises() -> None:
    with pytest.raises(ResponseParseError):
        parse_json_table({"metadata": [{"name": "ra"}]})


def test_invalid_json_raises_parse_error() -> None:
    with pytest.raises(ResponseParseError):
        parse_json_table(b"<html>not json</html>")


def test_empty_tap_result_is_empty_not_error() -> None:
    table = parse_json_table({"metadata": [{"name": "ra"}, {"name": "dec"}], "data": []})
    assert table.rows == [] and [c.name for c in table.columns] == ["ra", "dec"]


def test_nan_and_numpy_values_are_cleaned() -> None:
    table = parse_json_table({"metadata": [{"name": "a"}, {"name": "b"}], "data": [[float("nan"), np.float64(2.5)]]})
    assert table.rows[0] == {"a": None, "b": 2.5}
    assert type(table.rows[0]["b"]) is float


def test_legacy_list_of_dicts_still_supported() -> None:
    assert parse_json_records(json.dumps([{"ra": 1, "dec": 2}])) == [{"ra": 1, "dec": 2}]


# ---------------------------------------------------------------------------
# VOTable
# ---------------------------------------------------------------------------


def test_heasarc_binary_votable_keeps_units_and_ucds() -> None:
    table = parse_votable_table(body("3c273", "nvss"))
    assert table.format == "votable"
    assert table.rows[0]["name"] == "NVSS J122906+020305"
    assert table.column("ra").ucd.lower().startswith("pos.eq.ra")
    assert table.column("ra_error").unit == "s"  # seconds of time, NOT arcsec
    assert isinstance(table.rows[0]["ra"], float)


def test_irsa_votable_uses_field_names_not_col_ids() -> None:
    table = parse_votable_table(body("3c273", "twomass_psc"))
    assert "designation" in table.rows[0]
    assert not any(key.startswith("col_") for key in table.rows[0])
    assert table.rows[0]["designation"].strip() == "12290669+0203085"


def test_votable_query_status_error_raises_catalog_query_error() -> None:
    payload = (FIXTURES / "errors" / "heasarc_bad_column.0.body").read_bytes()
    assert votable_query_status(payload)[0] == "ERROR"
    with pytest.raises(CatalogQueryError, match="hard_hs"):
        parse_votable_table(payload)


def test_votable_overflow_is_reported_as_truncated() -> None:
    payload = b"""<?xml version="1.0"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE type="results">
<INFO name="QUERY_STATUS" value="OK"/>
<TABLE><FIELD name="ra" datatype="double" unit="deg" ucd="pos.eq.ra;meta.main"/>
<FIELD name="dec" datatype="double" unit="deg" ucd="pos.eq.dec;meta.main"/>
<DATA><TABLEDATA><TR><TD>1.0</TD><TD>2.0</TD></TR><TR><TD>3.0</TD><TD></TD></TR></TABLEDATA></DATA></TABLE>
<INFO name="QUERY_STATUS" value="OVERFLOW"/>
</RESOURCE></VOTABLE>"""
    table = parse_votable_table(payload)
    assert table.truncated
    assert table.rows[0] == {"ra": 1.0, "dec": 2.0}
    assert table.rows[1]["dec"] is None  # masked -> None
    assert parse_votable_records(payload)[0]["ra"] == 1.0


def test_votable_without_table_raises() -> None:
    payload = b'<?xml version="1.0"?><VOTABLE version="1.4"><RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/></RESOURCE></VOTABLE>'
    with pytest.raises(ResponseParseError):
        parse_votable_table(payload)


def test_query_status_attribute_order_and_content_attribute() -> None:
    assert votable_query_status(b'<INFO value="ERROR" name="QUERY_STATUS" content="boom"/>') == ("ERROR", "boom")
    assert votable_query_status(b"<VOTABLE/>") == (None, None)


# ---------------------------------------------------------------------------
# CSV / IPAC
# ---------------------------------------------------------------------------


def test_csv_skips_skyserver_comment_line() -> None:
    table = parse_csv_table((FIXTURES / "errors" / "sdss_conesearch_3c273.0.body").read_bytes())
    assert table.rows[0]["objid"] == "1237651735760142397"
    assert "objid" in [c.name for c in table.columns]
    assert parse_csv_records("#c\na,b\n1,2\n") == [{"a": "1", "b": "2"}]


def test_ipac_table_parse_with_units() -> None:
    payload = (
        "\\fixlen = T\n"
        "| ra        | dec       | designation | err_maj |\n"
        "| double    | double    | char        | double  |\n"
        "| deg       | deg       |             | arcsec  |\n"
        "|           |           |             | null    |\n"
        " 10.123456   -5.654321   J001          null\n"
    )
    table = parse_ipac_table(payload)
    assert table.rows[0]["designation"] == "J001"
    assert table.rows[0]["err_maj"] is None
    assert table.column("err_maj").unit == "arcsec"
    assert abs(parse_ipac_records(payload)[0]["ra"] - 10.123456) < 1e-9


def test_ipac_garbage_raises_parse_error() -> None:
    with pytest.raises(ResponseParseError):
        parse_ipac_table("this is not an ipac table")
