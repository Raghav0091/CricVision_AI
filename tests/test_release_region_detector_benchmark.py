from scripts.release_region_detector_benchmark import (
    _candidate_matches,
    _size_bucket,
)


def _annotation(point=(100, 100), size=8, visible=True):
    return {
        "ball_visible": visible,
        "ball_point": list(point) if point else None,
        "approx_ball_max_dimension_px": size,
    }


def test_candidate_matches_explicit_true_ball_point():
    candidate = {"center": [102, 101], "bbox_xyxy": [98, 97, 106, 105]}
    assert _candidate_matches(candidate, _annotation())


def test_distant_candidate_is_false_positive():
    candidate = {"center": [150, 150], "bbox_xyxy": [147, 147, 153, 153]}
    assert not _candidate_matches(candidate, _annotation())


def test_negative_frame_never_spatially_matches():
    candidate = {"center": [100, 100], "bbox_xyxy": [97, 97, 103, 103]}
    assert not _candidate_matches(candidate, _annotation(visible=False))


def test_size_buckets_follow_benchmark_contract():
    expected = {
        6: "<=6",
        7: "6-8",
        8: "6-8",
        9: "8-10",
        10: "8-10",
        11: "10-12",
        12: "10-12",
        13: "12-16",
        16: "12-16",
        17: ">16",
    }
    for size, bucket in expected.items():
        assert _size_bucket(_annotation(size=size)) == bucket
