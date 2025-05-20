mobile conversion script for FBCNN (a JPEG artifact denoiser)

how to use?

there's two ways

1. clone FBCNN
2. clone this repo on top of it (it should overwrite `models/fbcnn_network.py` with my patched one that is PyTorch Mobile compatible)
3. create a Python env

   you know the drill
5. `pip install -r requirements.txt`
6. download the original FNCNN models and move to `model_zoo`
7. then edit `convert_mobile.py` to the correct filenames and run

or use the premade `zip` file in releases for future-proofing in case the original repo or models are gone
