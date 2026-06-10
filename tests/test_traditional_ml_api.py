import numpy as np


def _toy_sample(offset):
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 40 + offset
    rgb[..., 1] = 60
    rgb[..., 2] = 80
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    rgb[mask == 1] = [230, 220, 80]
    fov = np.ones((16, 16), dtype=np.uint8)
    return {"rgb": rgb, "raw": rgb.copy(), "mask": mask, "fov": fov, "id": f"toy_{offset}"}


def test_traditional_models_train_svm_and_logistic_regression():
    from src.traditional_ml import predict_traditional, train_traditional_models

    samples = [_toy_sample(0), _toy_sample(10)]
    result = train_traditional_models(samples, seed=7, neg_per_pos=2)

    assert [m.name for m in result.models] == ["SVM", "Logistic Regression"]
    assert result.feature_names
    assert 0 <= result.candidate_recall <= 1

    for model in result.models:
        pred = predict_traditional(model, samples[0])
        assert pred.shape == samples[0]["mask"].shape
        assert pred.dtype == np.float32
        assert float(pred.min()) >= 0.0
        assert float(pred.max()) <= 1.0
