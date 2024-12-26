import torch
import torchmetrics
from torchmetrics.image.kid import KID
from torchmetrics.image.fid import FID
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from PIL import Image
import os
import tifffile as tiff
from tqdm import tqdm
from PIL import Image




def load_images_from_folder(folder_path, irange):
    images = []
    for filename in sorted(os.listdir(folder_path))[irange[0]:irange[1]]:
        if filename.endswith((".png")):
            img_path = os.path.join(folder_path, filename)
            #img = tiff.imread(img_path)
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
            images.append(img_tensor)
    images = torch.stack(images, 0)
    images = (images + 1) / 2
    images = (images * 255).type(torch.uint8)
    images = images.repeat(1, 3, 1, 1)
    return images.cuda()




def calculate_metrics(option, folder1, folder2, irange, subset_size=50):
    # Set seed for reproducibility
    torch.manual_seed(123)


    # Initialize metrics
    if option == 'kid':
        metrics = KID(subset_size=subset_size).cuda()
    elif option == 'fid':
        metrics = FID().cuda()
    elif option == 'lpips':
        metrics = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()


    # Load images
    for ix in tqdm(range(*irange)):
        imgs_dist1 = load_images_from_folder(folder1, irange=(ix, ix+irange[2]))[:, :, :, :]
        imgs_dist2 = load_images_from_folder(folder2, irange=(ix, ix+irange[2]))[:, :, :, :]

        #if option in ['kid', 'fid']:
        # Update metrics with images from both folders
        metrics.update(imgs_dist1, real=True)
        metrics.update(imgs_dist2, real=False)
        # Compute metrics
    try:
        metric_mean, metric_std = metrics.compute()
        print(f": {metric_mean:.4f} ± {metric_std:.4f}")
    except:
        metric_mean = metrics.compute()
        print(f": {metric_mean:.4f}")
    #else:
    #    print(metrics(imgs_dist1, imgs_dist2))
    return metric_mean


def quick_to_png(folder, destination):
    os.makedirs(destination, exist_ok=True)
    for filename in tqdm(sorted(os.listdir(folder))[:]):
        if filename.endswith((".tif")):
            img_path = os.path.join(folder, filename)
            img = tiff.imread(img_path)
            #img = np.transpose(img, (2, 1, 0))
            img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype('uint8')

            for z in range(img.shape[0]):
                slice = img[z, :, :]
                slice = Image.fromarray(slice)
                slice.save(destination + filename.replace('.tif', '') + '_' + str(z).zfill(3) + '.png')

root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_paper_figure/Figures/ablation_gan/present/'
quick_to_png(root + 'yzlinear/', root + 'png/yzlinear/')

if __name__ == '__main__':
    # All: 353: 216
    # Usage


    #root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/outputs_all/IsoLambda0/png/'

    root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_paper_figure/outputs_all/IsoLambda0/png/'

    root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_paper_figure/Figures/ablation_gan/present/png/'

    folder1 = root + 'xyori/'
    #folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/zya2d"
    folder2 = root + 'xz/'
    metric_mean = calculate_metrics('kid', folder1, folder2, irange=(0, 5520, 128))
    print(metric_mean)
    #kid  zx: 0.337, 0.1957  zy:  0.307, 0.208
    #fid  zx: 300, 198  zy:  280, 208
    #print(f"metrics: {metrics_mean:.4f} ± {metrics_std:.4f}")
