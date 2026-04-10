# To Do

1) Check that the gRPC client code supports TLS. Previously is was only connecting to the GPC server with TLS turned off (default). The following four tests should pass without error; 
    `python model_list.py --tls`
    `python model_list.py`
    `python generate.py --tls`
    `python generate.py`



## Main Methods
 - I2I
 - Moodboard 
 - ControlNet
    - Canny
    - Depth
    - Pose
    - Segmentation 
    - Inpaint
    - Scribble/Draw/LineArt
    - Instruct pix2pix
    - IPAdapter (SDXL)
    - IPAdapter Face (SDXL)

### Mrthod Add-Ons
 - Face Fix
 - Upscaler
