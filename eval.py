import os
import ml3d as _ml3d
import ml3d.tf as ml3d

def main():
    cfg_file = "ml3d/configs/randlanet_amsterdam3d.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)

    model = ml3d.models.RandLANet(**cfg.model)
    cfg.dataset['dataset_path'] = "datasets/Cyclomedia_pc_verified"
    dataset = _ml3d.datasets.Amsterdam3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

    # pretrained model
    ckpt_path = "logs/RandLANet_Amsterdam3D_tf/checkpoint/ckpt-1"

    # load the parameters.
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    test_split = dataset.get_split("val")
    data = test_split.get_data(0)

    # run inference on a single example.
    # returns dict with 'predict_labels' and 'predict_scores'.
    result = pipeline.run_inference(data)

    # evaluate performance on the test set; this will write logs to './logs'.
    pipeline.run_test()

if __name__ == "__main__":
    main()