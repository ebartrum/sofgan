import cv2, random
import torch, os, argparse
import numpy as np
from PIL import Image
from scipy.stats import norm
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm
import torch
import numpy as np

from modules.sof.utils.seg_sampler import FaceSegSampler
from modules.model_seg_input import Generator
from modules.BiSeNet import BiSeNet

from utils import *
from modules.model_seg_input import scatter as scatter_model
import sys,os

root = os.path.abspath('.')
os.chdir(root)
sys.path.append(root)

device = 'cuda'
torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IDList = [np.arange(17).tolist(),[0],[1,4,5,9,12],[15],[6,7,8,3],[11,13,14,16,10]]
# IDList = [[0],[1,4,5,9,12],[15],[2,3,6,7,8,10,11,13,14,16]]
groupName = ['Global','Background','Complexion','Hair','Eyes & Mouth','Wearings']
def scatter_to_mask(segementation, out_num=1,add_whole=True,add_flip=False,region=None):
    segementation = scatter_model(segementation)
    masks = []

    if None == region:
        if add_whole:
            mask = torch.sum(segementation, dim=1, keepdim=True).clamp(0.0, 1.0)
            masks.append(torch.cat((mask, 1.0 - mask), dim=1))
        if add_flip:
            masks.append(torch.cat((1.0 - mask, mask), dim=1))


        for i in range(out_num - add_whole - add_flip):
            idList = IDList[i]
            mask = torch.sum(segementation[:, idList], dim=1, keepdim=True).clamp(0.0, 1.0)
            masks.append(torch.cat((1.0 - mask, mask), dim=1))
    else:
        for item in region:
            idList = IDList[item]
            mask = torch.sum(segementation[:, idList], dim=1, keepdim=True).clamp(0.0, 1.0)
            masks.append(torch.cat((1.0 - mask, mask), dim=1))
    masks = torch.cat(masks, dim=0)
    return masks

def make_noise(batch, styles_dim, style_repeat, latent_dim, n_noise, device):
    noises = torch.randn(n_noise, batch, styles_dim, latent_dim, device=device).repeat(1, 1, style_repeat, 1)
    return noises

def mixing_noise(batch, latent_dim, prob, device, unbine=True):
    n_noise = 1
    style_dim = 2 if random.random() < prob else 1
    style_repeat = 2 // style_dim  # if prob>0 else 1
    styles = make_noise(batch, style_dim, style_repeat, latent_dim, n_noise, device)
    return styles.unbind(0) if unbine else styles

def sample_styles_with_miou(seg_label, num_style, mixstyle=0, truncation=0.9, batch_size=4, descending=False):
    times = 0
    in_batch = seg_label.shape[0]
    if in_batch == 1:
        batch = batch_size
        seg_label = seg_label.repeat(batch, 1, 1, 1)
    else:
        batch = in_batch

    with torch.no_grad():
        styles_miou, count, mious = [], 0, []
        while count < num_style:
            styles = mixing_noise(batch // in_batch, args.latent, mixstyle, device, unbine=False)
            styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=truncation)
            styles = torch.cat(styles, dim=0)
            w_latent = generator.style_map([styles], to_w_space=False)

            if in_batch > 1:
                w_latent = w_latent.repeat(batch, 1, 1)

            img, _, _, _ = generator(w_latent, return_latents=False, condition_img=seg_label, input_is_latent=True,
                                     noise=noise)
            img = img.clamp(-1.0, 1.0)
            img = F.interpolate(img, size=(512, 512), mode='bilinear')

            segmap = bisNet(img)[0]
            segmap = F.interpolate(segmap, size=seg_label.shape[2:], mode='bilinear')
            segmap = id_remap(torch.argmax(segmap, dim=1, keepdim=True))

            thread = 0.46
            if times > 15:
                thread = 0.42
            if times > 20:
                thread = 0.35
            if times > 30:
                thread = 0.

            miou = mIOU(segmap, seg_label)
            miou = miou.min() if in_batch > 1 else miou
            mask = (miou > thread).tolist()

            times += 1
            if np.sum(mask) == 0:
                continue

            if in_batch > 1 and mask:
                mious.append(miou.view(-1, 1))
                styles_miou.append(w_latent[[0]])
                count += 1
            else:
                mious.append(miou[mask])
                if len(mask) == w_latent.shape[0]:
                    styles_miou.append(w_latent[mask])
                else:
                    styles_miou.append(
                        w_latent.view(-1, 2, w_latent.shape[-2], w_latent.shape[-1])[mask])  # old need this
                count += np.sum(mask)

    mious = torch.cat(mious, dim=0).view(-1)
    mious, indices = torch.sort(mious, descending=descending)
    styles_miou = torch.cat(styles_miou, dim=0)[indices]
    return styles_miou[:num_style]

def initFaceParsing(n_classes=20):
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load('./ckpts/segNet-20Class.pth'))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
    return net, to_tensor

def parsing_img(bisNet, image, to_tensor, argmax=True):
    with torch.no_grad():
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0).cuda()
        segmap = bisNet(img)[0]
        if argmax:
            segmap = segmap.argmax(1, keepdim=True)
        segmap = id_remap(segmap)
    return img, segmap

def auto_crop_img(image, detector=None, inv_pad=2):
    if detector is None:
        detector = dlib.get_frontal_face_detector()

    dets = detector(image, 1)
    h, w = image.shape[:2]

    faces = []
    for i, d in enumerate(dets):
        left, right, top, bottom = d.left(), d.right(), d.top(), d.bottom()
        width_crop = right - left
        pad = min(w - right, left, top, h - bottom, width_crop // inv_pad)

        top = max(top - int(pad * 1.5), 0)
        faces.append(image[top:top + width_crop + 2 * pad, left - pad:right + pad])
    return faces

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str)
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-batch_size', type=int,default=4)
parser.add_argument('--resolution', type=int, default=1024)
parser.add_argument('--nrows', type=int, default=6)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--channel_multiplier', type=int, default=2)
parser.add_argument('--with_rgb_input', action='store_true')
parser.add_argument('--with_local_style', action='store_true')
parser.add_argument('--condition_dim', type=int, default=0)
parser.add_argument('--styles_path', type=str, default=None)
parser.add_argument('--MODE', type=int, default=0)
parser.add_argument('--miou_filter', action='store_true')
parser.add_argument('--truncation', type=float, default=0.7)
parser.add_argument('--with_seg_fc', action='store_true')

cmd = f'-i ./dataset/video -o ./result/mv-obama/ \
--ckpt ./ckpts/generator.pt \
--resolution 1024  --MODE 2 --miou_filter --truncation 0.7'
args = parser.parse_args(cmd.split())

# define networks
args.latent = 512
args.n_mlp = 8
args.condition_path = args.input
generator = Generator(args).eval().to(device)

ckpt = torch.load(args.ckpt)
generator.load_state_dict(ckpt['g_ema'])

batch_size = 4
latent_av = cal_av(generator, batch_size, args.latent)

# face parser
bisNet, to_tensor = initFaceParsing()

del ckpt
torch.cuda.empty_cache()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

"""
img_path = './example/Harry.jpg'# path to the source image folder
save_path = './example/test.png'
auto_crop = False # you need to center crop the image if you are using your own photos; please set false if image comes from FFHQ or CelebA
miou_filter = True # set true if you want to filter style with the miou
n_styles = 3
resolution_vis = 1024 # image resolution to save 
save_as_video = True

with torch.no_grad():

    noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]

    img = Image.open(img_path).convert('RGB')
    if auto_crop:
        import dlib
        faces = auto_crop_img(np.array(img))
        img = Image.fromarray(faces[0])
        
    img, seg_label = parsing_img(bisNet, img.resize((512, 512)), to_tensor)
    seg_label_rgb = vis_condition_img(seg_label)
    seg_label_rgb = F.interpolate(seg_label_rgb, (args.resolution, args.resolution), mode='bilinear', align_corners=True)

    try:
        tqdm._instances.clear() 
    except Exception:     
        pass
        
    if not save_as_video:
        if miou_filter:
            w_latent = sample_styles_with_miou(seg_label, n_styles * 2, mixstyle=mixstyle,
                                           truncation=args.truncation, batch_size=args.batch_size)
        else:
            mixstyle = 0.0
            styles = mixing_noise(n_styles, args.latent, mixstyle, device, unbine=False)
            styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
            styles = torch.cat(styles, dim=0)
            w_latent = generator.style_map([styles], to_w_space=False)
            w_latent = w_latent.view(-1,2,w_latent.shape[-2],w_latent.shape[-1])
    
        result = [F.interpolate(img.cpu(),(resolution_vis,resolution_vis), mode='bilinear', align_corners=True)]
        for j in tqdm(range(n_styles)):
            fake_img, _, _, _ = generator(w_latent[j], return_latents=False, condition_img=seg_label, \
                                          input_is_latent=True, noise=noise)
            result.append(F.interpolate(fake_img.detach().cpu().clamp(-1.0, 1.0),(resolution_vis,resolution_vis), 
                                        mode='bilinear', align_corners=True))

        result = torch.cat(result, dim=0)
        utils.save_image(result, save_path,nrow=n_styles+1,normalize=True,range=(-1, 1),padding = 2)
    else:
        nrows,ncols = 1, 2
        width_pad, height_pad = 2 * (ncols + 1), 2 * (nrows + 1) 
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        print(f'{save_path[:-4]}.mp4')
        out = cv2.VideoWriter(f'{save_path[:-4]}.mp4', fourcc,
                              20, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))

        img = F.interpolate(img.cpu(),(resolution_vis,resolution_vis), mode='bilinear', align_corners=True)
        w_latents = sample_styles_with_miou(seg_label, n_styles, mixstyle=0.0, truncation=args.truncation, batch_size=args.batch_size,descending=True)[:,0]
        style_masks = scatter_to_mask(seg_label, len(groupName), add_flip=False, add_whole=False)
        
        w_latent_nexts = []
        for i_style in tqdm(range(len(groupName))):

            regions = list(range(n_styles)) + [0]

            for j,frame in enumerate(range(1,len(regions))):

                if 0 == regions[frame - 1]: # first style
                    w_latent_last, w_latent_next = w_latents[:1], w_latents[[frame]]
                elif 0 == regions[frame]:# last style
                    w_latent_last, w_latent_next = w_latent_next.clone(), w_latents[:1]
                else:
                    w_latent_last = w_latent_next.clone()
                    w_latent_next = w_latents[[frame]].clone()

                frame_sub_count = 40 if i_style<4 else 30
                cdf_scale = 1.0 / (1.0 - norm.cdf(-frame_sub_count // 2, 0, 6) * 2)
                for frame_sub in range(-frame_sub_count // 2, frame_sub_count // 2 + 1):

                    weight = (norm.cdf(frame_sub, 0, 6) - norm.cdf(-frame_sub_count // 2, 0, 6)) * cdf_scale

                    w_latent_current = (1.0 - weight) * w_latent_last + weight * w_latent_next
                    w_latent_current = torch.cat((w_latents[:1],w_latent_current),dim=0)


                    # first row
                    result = [img]
                    w_latent_current_in = w_latent_current.view(-1, 18, 512)
                    fake_img, _, _, _ = generator(w_latent_current_in, return_latents=False,
                                                  condition_img=seg_label, \
                                                  input_is_latent=True, noise=noise,
                                                  style_mask=style_masks[[i_style]])
                    result.append(F.interpolate(fake_img.detach().cpu().clamp(-1.0, 1.0),(resolution_vis,resolution_vis)
                                               , mode='bilinear', align_corners=True))

                    result = torch.cat(result, dim=0)
                    result = (utils.make_grid(result, nrow=ncols) + 1) / 2 * 255
                    result = (result.detach().numpy()[::-1]).transpose((1, 2, 0))
                    out.write(result.astype('uint8'))
        out.release()
"""



"""
# # video style transfer
video_path = './example/faceCap.avi'# path to the source image folder
save_path = './example/faceCap-restyle.mp4'
auto_crop = False # you need to center crop the image if you are using your own photos; please set false if image is from FFHQ or CelebA
resolution_vis = 512 # image resolution to save 
save_as_video = True

cap = cv2.VideoCapture(video_path)
nrows,ncols = 2, 2
width_pad, height_pad = 2 * (ncols + 1), 2 * (nrows + 1) 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(f'{save_path[:-4]}.mp4', fourcc,
                      20, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))

with torch.no_grad():

    noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
    
    mixstyle = 0.0
    styles = mixing_noise(nrows*ncols, args.latent, mixstyle, device, unbine=False)
    styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
    styles = torch.cat(styles, dim=0)
    w_latent = generator.style_map([styles], to_w_space=False)
    w_latent = w_latent.view(-1,2,w_latent.shape[-2],w_latent.shape[-1])

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, img = cap.read()
    
    try:
        tqdm._instances.clear() 
    except Exception:     
        pass
    for _ in tqdm(range(length-1)):

        if auto_crop:
            import dlib
            faces = auto_crop_img(img[...,::-1])#bgr -> rgb
            img = Image.fromarray(faces[0])
        else:
            img = Image.fromarray(img[...,::-1])

        # you may need a facial landmark detector to remove the jitter on the eyes region
        # we provide a more stable video parser, you can find it in the readme.
        img, seg_label = parsing_img(bisNet, img.resize((512, 512)), to_tensor)
        seg_label_rgb = vis_condition_img(seg_label)
        seg_label_rgb = F.interpolate(seg_label_rgb, (args.resolution, args.resolution), mode='bilinear', align_corners=True)
        

        result = [F.interpolate(img.cpu(),(resolution_vis,resolution_vis))]
        for j in range(nrows*ncols-1):
            fake_img, _, _, _ = generator(w_latent[j], return_latents=False, condition_img=seg_label, \
                                          input_is_latent=True, noise=noise)
            result.append(F.interpolate(fake_img.detach().cpu().clamp(-1.0, 1.0),(resolution_vis,resolution_vis)))

        result = torch.cat(result, dim=0)
        result = (utils.make_grid(result, nrow=nrows) + 1) / 2 * 255
        result = result.numpy()[::-1].transpose((1, 2, 0)).astype('uint8')
        out.write(result)
        result = []
        success, img = cap.read()
out.release()
"""

inference_mode = "appearance"
# inference_mode = "azimuth"

img_size = 128
radius = 4.5
resolution_vis = 512 # image resolution to save 

if inference_mode == "azimuth":
    num_poses = 3
    seg_sampler = FaceSegSampler(
        model_path='./ckpts/epoch_0250_iter_050000.pth', 
        img_size=512, 
        sample_mode="azimuth",
        sample_radius=radius,
        max_batch_size=num_poses
        )

    nrows,ncols = 2, 2
    width_pad, height_pad = 2 * (ncols + 1), 2 * (nrows + 1) 
    n_feames = num_poses
    num_objs = 1000

    # sampling poses
    look_at = np.asarray([0, 0.1, 0.0])
    cam_center =  np.asarray([0, 0.1, 4.5])

    # generate images
    with torch.no_grad():
        for obj_id in range(num_objs):
            # sampling instance embedding (Controls shape)
            smp_ins = torch.from_numpy(seg_sampler.gmm.sample(1)[0]).float()
            smp_poses, nocs_maps = seg_sampler.sample_pose(
                cam_center, look_at, 
                num_samples=n_feames, emb=smp_ins)

            save_dir = f'./eval/{inference_mode}/obj_{obj_id}'
            os.makedirs(save_dir, exist_ok=True)
            seg_label = id_remap(torch.from_numpy(smp_poses[:1]).float()).to(device)
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            w_latent = sample_styles_with_miou(
                    seg_label, 1, mixstyle=0.0, truncation=args.truncation, batch_size=args.batch_size,descending=True)[0]

            try:
                tqdm._instances.clear() 
            except Exception:     
                pass
            for i, seg_label in enumerate(tqdm(smp_poses)):
                seg_label = id_remap(torch.from_numpy(seg_label).float()[None,None]).to(device)
                fake_img, _, _, _ = generator(  w_latent, return_latents=False,
                                                condition_img=seg_label, \
                                                input_is_latent=True, noise=noise)
                fake_img_out = F.interpolate(fake_img.detach().cpu().clamp(-1.0, 1.0),
                        (resolution_vis,resolution_vis)).squeeze(0)

                fake_img_out = (fake_img_out + 1)/ 2 * 255
                fake_img_out = fake_img_out.numpy().transpose((1, 2, 0)).astype('uint8')

                out = Image.fromarray(fake_img_out)
                filename = f"{i}.png"
                full_path = os.path.join(save_dir, filename)
                out.save(full_path)

                nocs_map_out = nocs_maps[i].cpu()
                world_depth_out = nocs_map_out[:, :, 2]
                world_depth_filename = f"world_depth_{i}.pt"
                full_world_depth_path = os.path.join(save_dir, world_depth_filename)
                torch.save(world_depth_out, full_world_depth_path)

if inference_mode == "appearance":
    num_objs = 5
    num_appearances = 5
    frontal_seg_sampler = FaceSegSampler(
        model_path='./ckpts/epoch_0250_iter_050000.pth', 
        img_size=512, 
        sample_mode="frontal",
        sample_radius=radius,
        max_batch_size=2
        )

    n_feames = num_appearances

    # sampling poses
    look_at = np.asarray([0, 0.1, 0.0])
    cam_center =  np.asarray([0, 0.1, 4.5])

    # generate images
    with torch.no_grad():
        for obj_id in range(num_objs):
            # sampling instance embedding (Controls shape)
            smp_ins = torch.from_numpy(frontal_seg_sampler.gmm.sample(1)[0]).float()
            smp_poses, _ = frontal_seg_sampler.sample_pose(
                cam_center, look_at, 
                num_samples=n_feames, emb=smp_ins)
            smp_poses = smp_poses[[0]] # Only need 1 frontal pose

            save_dir = f'./eval/{inference_mode}/obj_{obj_id}'
            os.makedirs(save_dir, exist_ok=True)
            seg_label = id_remap(torch.from_numpy(smp_poses[:1]).float()).to(device)
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            w_latent = sample_styles_with_miou(
                    seg_label, 1, mixstyle=0.0, truncation=args.truncation, batch_size=args.batch_size,descending=True)[0]

            try:
                tqdm._instances.clear() 
            except Exception:     
                pass
            seg_label = smp_poses[0]
            seg_label = id_remap(torch.from_numpy(seg_label).float()[None,None]).to(device)
            for i in range(num_appearances):
                new_w_latent = sample_styles_with_miou(
                        seg_label, 1, mixstyle=0.0, truncation=args.truncation, batch_size=args.batch_size,descending=True)[0]
                w_latent = new_w_latent

                fake_img, _, _, _ = generator(  w_latent, return_latents=False,
                                                condition_img=seg_label, \
                                                input_is_latent=True, noise=noise)
                fake_img_out = F.interpolate(fake_img.detach().cpu().clamp(-1.0, 1.0),
                        (resolution_vis,resolution_vis)).squeeze(0)

                fake_img_out = (fake_img_out + 1)/ 2 * 255
                fake_img_out = fake_img_out.numpy().transpose((1, 2, 0)).astype('uint8')

                out = Image.fromarray(fake_img_out)
                filename = f"{i}.png"
                full_path = os.path.join(save_dir, filename)
                out.save(full_path)
