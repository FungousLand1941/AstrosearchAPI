# Telescope signals, Mantis representations, and novelty triage

This design turns Vatsal's proposal into a testable pipeline. It accepts calibrated light curves and spectra, creates a fixed representation, cross-references each observation against known objects, and emits a typed vector row for a dedicated Mantis map. It calls unusual observations **novelty-review candidates**, never discoveries.

## Why this needs a separate ingestion path

Catalog rows and text summaries describe known objects. A light curve or spectrum is an ordered measurement with uncertainty, missing samples, instrument artifacts, and an axis with physical units. Flattening these arrays into prose destroys the evidence needed for comparison. The signal path therefore preserves the original observation and constructs a versioned numeric representation beside it.

The baseline follows three authoritative data constraints:

- TESS light-curve products contain time, flux, flux uncertainty, and quality values, and SAP and PDCSAP have different scientific uses. See the [NASA TESS light-curve documentation](https://heasarc.gsfc.nasa.gov/docs/tess/LightCurveFile-Object-Tutorial.html) and [MAST product description](https://archive.stsci.edu/missions-and-data/tess/data-products).
- NEOWISE single-exposure photometry includes band-specific uncertainties and contamination flags; low-SNR magnitudes can be upper limits rather than detections. See the [IRSA column definitions](https://irsa.ipac.caltech.edu/data/WISE/docs/release/NEOWISE/expsup/sec2_1a.html) and [cautionary notes](https://irsa.ipac.caltech.edu/data/WISE/docs/release/NEOWISE/expsup/sec2_3.html).
- Spectra and time series need coordinate, uncertainty, and provenance metadata. The [IVOA Spectrum Data Model 1.2](https://www.ivoa.net/documents/SpectrumDM/) provides the interoperability baseline.

## Implemented flow

```text
telescope/archive delivery
        │
        ▼
mission adapter: FITS/table → canonical observation
        │  axis, values, uncertainties, quality flags, ICRS position, provenance
        ▼
quality gate and robust normalization
        │  bad/non-finite samples removed; usable fraction retained
        ▼
astrosearch-signal-v1 (48 dimensions)
        │  12 robust statistics + 32 shape bins + 4 autocorrelations
        ▼
two independent comparisons
        ├── sky/epoch cross-match against known catalog objects
        └── cosine distance against same-modality reference signals
        ▼
triage status + immutable evidence bundle + Mantis vector row
```

`representations.py` implements the deterministic baseline. It sorts the physical axis, rejects unusable input, applies quality flags, uses uncertainties in the trend estimate, robustly scales the signal, and records its usable-sample fraction. Spectral binning uses log wavelength when all wavelengths are positive. Every vector is unit-normalized and tagged `astrosearch-signal-v1`; incompatible versions must not be compared.

`signal_pipeline.py` accepts newline-delimited JSON deliveries, rejects conflicting duplicate observation IDs, produces an immutable fingerprinted snapshot, and writes:

- `delivery.json`: original canonical observations and thresholds;
- `cross-reference-results.json`: ranked evidence and review requirements;
- `manifest.json`: counts, representation version, and fingerprint;
- `mantis-signals.csv`: typed metadata and a 48-value vector column.

Repeated identical deliveries return `changed: false`. A reused observation ID with different content is rejected instead of silently overwriting provenance.

Every reference vector must include its `modality` and `representation_version`. Vectors with a different modality or version remain usable as positional evidence but are never compared numerically.

## Canonical observation contract

```json
{
  "observation_id": "tess-sector-42-tic-123",
  "object_id": "TIC 123",
  "modality": "light_curve",
  "ra_deg": 123.4,
  "dec_deg": -20.5,
  "axis": [0.0, 0.02, 0.04],
  "values": [1.0, 0.999, 1.001],
  "uncertainties": [0.001, 0.001, 0.001],
  "quality": [0, 0, 0],
  "instrument": "TESS",
  "band": "TESS",
  "observed_at": "2026-09-21T00:00:00Z",
  "source_url": "https://archive.example/product",
  "provenance": {
    "archive": "MAST",
    "product_id": "...",
    "pipeline": "SPOC",
    "pipeline_version": "...",
    "axis_unit": "BTJD day",
    "value_unit": "electron / s",
    "value_kind": "PDCSAP_FLUX"
  }
}
```

The adapter, rather than the representation function, owns mission-specific decisions. For TESS it must preserve the sector and distinguish SAP from PDCSAP. For NEOWISE it must preserve W1/W2, upper limits, `cc_flags`, SNR, reduced chi-squared, frame identifiers, and the observation epoch. Spectra must preserve wavelength frame, flux convention, spectral resolution, calibration level, redshift/rest-frame treatment, and masks.

`tess_adapter.py` implements the first mission adapter for SPOC light-curve FITS files. It supports SAP and PDCSAP explicitly and preserves the full quality array, uncertainty array, sector, detector, pipeline version, units, archive URL, and original product name.

## Cross-reference states

| Status | Meaning | Next action |
|---|---|---|
| `matched` | One positional candidate also has a similar signal representation | Verify identity/provenance, then associate |
| `ambiguous` | Several positional candidates have similar representations | Resolve blending, proper motion, and source attribution |
| `position_only_review` | Position overlaps a known source, but signal evidence is absent or inconsistent | Inspect pixels, epoch, and catalog data |
| `unmatched` | No positional match; representation is not sufficiently anomalous | Retain for future reference coverage |
| `insufficient_reference_coverage` | No comparable reference vectors exist | Do not score novelty |
| `novelty_review_candidate` | No positional match, adequate data quality, and distant from available reference signals | Run the discovery validation protocol |

The thresholds are explicit inputs and recorded in every result. They are placeholders until calibrated separately for each instrument, modality, processing version, sky region, magnitude/SNR regime, and science objective.

## Discovery validation protocol

A Mantis outlier is a prioritization result. Before scientific communication, a candidate needs all of the following:

1. Inspect pixels, masks, background, saturation, cosmic rays, persistence, diffraction features, blending, and pipeline diagnostics.
2. Repeat the positional cross-match at the observation epoch, propagating proper motion and checking solar-system ephemerides.
3. Compare only against compatible representations: same modality, calibrated units, instrument response, processing lineage, and adequate SNR.
4. Confirm the signal in another cadence, visit, sector, exposure, or preferably another instrument.
5. Estimate a false-positive rate on held-out known objects and injected signals before choosing an alert threshold.
6. Record a human review decision with the exact data, code version, thresholds, and rejected alternatives.
7. Use the relevant community reporting and validation process. The software continues to report `discovery_claim: false`.

## Representation work for the expert group

The shipped 48-dimensional vector is an interpretable baseline and integration contract. Representation specialists should compare it against modality-specific alternatives under leakage-safe evaluation:

- light curves: periodograms, wavelets, Gaussian-process residuals, phase-folded transit representations, self-supervised sequence encoders, and models that retain irregular cadence;
- spectra/infrared: continuum-normalized line features, resolution-aware resampling, uncertainty-aware spectral encoders, and joint photometry/spectrum embeddings;
- images/cubes: PSF-aware cutouts, difference images, segmentation/context channels, and instrument-specific artifact embeddings;
- multimodal fusion: late fusion of sky, time, spectrum, image, and catalog evidence with missing-modality masks rather than concatenating uncalibrated values.

Evaluation must split by object and observing campaign, preserve rare classes in validation, measure retrieval quality and anomaly false-positive rate, test cross-instrument drift, and include artifact challenge sets. A learned vector must publish its training-set fingerprint, code/model checksum, preprocessing contract, dimensionality, distance metric, and calibration report before it replaces `astrosearch-signal-v1`.

## Running it

```bash
astrosearch-signals telescope-delivery.jsonl --references known-signal-references.json --state-dir astronomy-data
```

The API exposes the same single-observation logic at `POST /api/v1/signals/cross-reference`.

For Mantis, create a dedicated signal map from `mantis-signals.csv`. Type `representation` as a vector; use `title` as title; `summary` as semantic text; `catalog_id`, `object_id`, `modality`, `instrument`, `band`, `triage_status`, and `representation_version` as categories; the count/fraction/distance and coordinates as numeric fields; `observed_at` as date; and `source_url` as a link. Keep this map distinct from the catalog and sky maps so representation versions and update cadence remain auditable.

## Meeting decisions needed

The engineering path is ready. The representation group should decide the first telescope product and science target, the canonical calibration level, the identity radius/epoch policy, the reference corpus, the acceptable false-positive budget, and who may promote a Mantis outlier into an observing follow-up queue. A practical pilot is one TESS sector with known variable stars, known planet transits, injected signals, and detector artifacts; it is bounded, labeled, and directly exercises the whole flow.
