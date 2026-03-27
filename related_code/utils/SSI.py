import torch
import torch.fft
import numpy as np

def extract_ampl_phase(fft_im):
    fft_amp = fft_im.abs()
    fft_pha = fft_im.angle()
    
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.01):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*L)).astype(int)
    
    amp_src = torch.fft.fftshift(amp_src, dim=(-2, -1))
    amp_trg = torch.fft.fftshift(amp_trg, dim=(-2, -1))

    c_h = h // 2
    c_w = w // 2

    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1
    
    amp_src[:, :, h1:h2, w1:w2] = amp_trg[:, :, h1:h2, w1:w2]
    amp_src = torch.fft.ifftshift(amp_src, dim=(-2, -1))

    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.01):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))

    return a_src


def FDA_source_to_target(src_img, trg_img, L=0.01):
    # input: src_img, trg_img (Batch, Channel, Height, Width)

    fft_src = torch.fft.fft2(src_img.clone(), dim=(-2, -1))
    fft_trg = torch.fft.fft2(trg_img.clone(), dim=(-2, -1))

    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, pha_trg = extract_ampl_phase(fft_trg)

    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    fft_src_ = torch.polar(amp_src_, pha_src)

    src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1)).real

    return src_in_trg


def FDA_source_to_target_np( src_img, trg_img, L=0.01):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg
