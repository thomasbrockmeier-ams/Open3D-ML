import os
import ml3d as _ml3d
import ml3d.tf as ml3d

def main():
    cfg_file = "ml3d/configs/randlanet_amsterdam3d.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)

    model = ml3d.models.RandLANet(**cfg.model)
    cfg.dataset['dataset_path'] = "datasets/Cyclomedia_pc_verified"
    dataset = _ml3d.datasets.Amsterdam3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = ml3d.pipelines.SemanticSegmentation(model=model, dataset=dataset, max_epoch=200, batch_size=1, device='gpu')

    ckpt_folder = "./logs/"
    os.makedirs(ckpt_folder, exist_ok=True)

    pipeline.cfg_tb = {
        "readme": "readme",
        "cmd_line": "cmd_line",
        "dataset": "Amsterdam3D",
        "model": "RandLaNet",
        "pipeline": "Default Pipeline",
    }

    pipeline.run_train()

if __name__ == "__main__":
    main()