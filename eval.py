import tqdm
import numpy as np
import norbert
import museval
import scipy.signal, scipy
import os
import json

sr = 44100
sources = ['mixture', 'drums', 'bass', 'other', 'vocals']

def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio

def separate(audio, targets, model_name='umxhq', niter=1, softmask=False, alpha=1.0, residual_model=False, device="cpu"):
    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []

    for j, target in enumerate(tqdm.tqdm(targets)):
        unmix_target = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', target = target)
        Vj = unmix_target(audio_torch).cpu().detach().numpy()
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    X = unmix_target.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(targets) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=unmix_target.stft.n_fft,
            n_hopsize=unmix_target.stft.n_hop
        )
        estimates[name] = audio_hat.T

    return estimates


def separate_and_evaluate(track, targets, device, output_dir):
    estimates = separate(audio = track.audio, targets = targets, device = device)
    scores = museval.eval_mus_track(track, estimates, output_dir = output_dir)
    return scores

def getMedianMetrics(filename, metric):
    with open(filename) as f:
        data = json.load(f)
        vocalsEval = data['targets'][0]['frames']
        vocalsMetrics  = [t['metrics'][metric] for t in vocalsEval]
    return np.median(vocalsMetrics)


if __init__ == "__main__":
    for idx, track in enumerate(mus):
        if idx % (len(mus) // 20) == 0:
            print ("Process : {} / {}".format(idx, len(mus)))
        separate_and_evaluate(track = track, targets = ['vocals'], device = "cpu", output_dir = 'eval/')
    
    SDRs = []
    for file in EVALFILES:
        SDRs.append(getMedianMetrics(file, 'SDR'))
    print ("median frame SDR : {}".format(np.median(SDRs)))



