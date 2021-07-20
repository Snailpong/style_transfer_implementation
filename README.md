# style_transfer_implementation
PyTorch implementation of style transfer (landscape cartoonization) models

## Models

- [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
- [CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)
- [AnimeGAN](https://link.springer.com/chapter/10.1007/978-981-15-5577-0_18)


## Dependencies

- Pytorch
- torchvision
- numpy
- PIL
- OpenCV
- tqdm
- click


## Usage

1. Download dataset
- CycleGAN: [Link](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)
- CartoonGAN, AnimeGAN: [Link](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)

2. Place data

e.g.
```
.
└── data
    ├── summer2winter_yosemite
    |   ├── trainA
    |   ├── trainB
    |   ├── testA
    |   └── testB
    └── cartoon_dataset
        ├── photo
        |   ├── 0.jpg
        |   └── ...
        ├── cartoon
        |   ├── 0_1.jpg
        |   └── ...
        ├── cartoon_smoothed
        |   ├── 0_1.jpg
        |   └── ...
        └── val
            ├── 1.jpg
            └── ...
```

3. Train

- CycleGAN: ``` python train_cyclegan.py```
- CartoonGAN: ``` python train_cartoongan.py```
- AnimeGAN: ``` python train_animegan.py```

- arguments
    - dataset_type (only for CycleGAN): folder name to use for dataset (e.g. summer2winter_yosemite)
    - load_model: ```True```/```False```
    - cuda_visible: ```CUDA_VISIBLE_DEVICES``` (e.g. 1)


4. Test
- CycleGAN: ``` python test_cyclegan.py```
- CartoonGAN: ``` python test_cartoongan.py```
- AnimeGAN: ``` python test_cartoongan.py --model_name=animegan```

- arguments
    - dataset_type (only for CycleGAN)
    - model_type (only for CycleGAN): ```x2y```/```y2x```
    - image_path: folder path to convert the images
    - cuda_visible
    - model_name (only for CartoonGAN): ```cartoongan```/```animegan```
    - is_crop (only for CartoonGAN): ```True```/```False```. crop and resize image to (256, 256)


## Results

- We trained the models with Shinkai style

<table style="text-align: center">
<tr><td>Photo</td><td>CartoonGAN</td><td>AnimeGAN</td></tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325567-bf2c0b10-5235-44df-a929-d05d2ea89253.jpg" alt="2014-09-08 05_31_48"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324701-04ec98ef-7ec5-409a-ab65-44ffb47df59a.jpg" alt="2014-09-08 05_31_48"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325115-f5ec1282-475c-45c2-9e14-32a0fe0e9f74.jpg" alt="2014-09-08 05_31_48"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325570-00bde8c7-ff74-48a1-af01-ec836ef4fc1b.jpg" alt="2014-12-07 05_00_46"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324708-29c68ac0-ee7e-4514-b216-dccc543d2668.jpg" alt="2014-12-07 05_00_46"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325123-83cbf68c-4103-4622-83ab-0d7b12a85c62.jpg" alt="2014-12-07 05_00_46"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325590-6771f3eb-fdb1-4d48-9239-c538ff262ac6.jpg" alt="11"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324737-ed7545cc-0802-49ef-8ae2-fe0f76096196.jpg" alt="11"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325133-d86e2be4-5cb3-4620-9aaf-a04052f108bd.jpg" alt="11"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325593-9b123c39-5265-4508-9334-069ceaefdb6e.jpg" alt="16"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324747-413b3b7a-81c2-4e06-861f-36b96e52a00b.jpg" alt="16"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325141-6de4a3e5-4e72-4836-933c-ed1a08293e3d.jpg" alt="16"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325600-a2eeffb0-9299-4a9c-9373-1eb49de8dabf.jpg" alt="17"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324759-40db15c3-8a24-4ff9-912f-f242c6478bd3.jpg" alt="17"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325146-26329da1-ddfc-4904-81c8-97e7dca277fb.jpg" alt="17"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325607-a939239b-6d91-49c6-93de-dc8556ae6770.jpg" alt="20"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324773-8ea42552-abab-49c4-b405-a61a3944b543.jpg" alt="20"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325154-9d609f6e-9a37-4f42-98d3-eb6d65f3f356.jpg" alt="20"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325621-6ed08842-b0a8-4b38-bca3-3423695fb85b.jpg" alt="30"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324796-eb0df69f-287e-4f89-b042-c5f78629cec5.jpg" alt="30"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325164-a284fc6f-2c38-4859-a13d-faa9da980772.jpg" alt="30"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/126325632-b72c0a43-e8c0-46b7-8322-940733b3be55.jpg" alt="55"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126324808-3ca5d3c2-a728-4fb1-a2f4-e84b43868d6a.jpg" alt="55"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/126325173-1e3ea28c-1826-42f3-9c41-99d28ffa88b8.jpg" alt="55"></td>
</tr>

</table>

## Observation & Discussion

- AnimeGAN kept the original texture and color than CartoonGAN.
- CartoonGAN made good use of the unique black edge of cartoons. 
- CartoonGAN and AnimeGAN were close to the texture of TVD/Movie, respectively.
- AnimeGAN did not reduce the Discriminator adversial loss from certain point.
- In CartoonGAN, the color expressions changed as the epoch increases, and was unified for all generated images.
- The performance was better when using generated cartoon data by cropping the high-resolution images than resizing them.



## Code Reference

- <https://github.com/TobiasSunderdiek/cartoon-gan>
- <https://github.com/mnmjh1215/diya-CartoonGAN>
- <https://github.com/TachibanaYoshino/AnimeGAN>