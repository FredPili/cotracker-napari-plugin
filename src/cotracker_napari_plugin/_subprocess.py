import argparse
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_path')
    parser.add_argument('queries_path')
    parser.add_argument('output_path')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    
    try:
        import torch
        
        frames = np.load(args.frames_path, mmap_mode='r')
        queries = np.load(args.queries_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {device}", file=sys.stderr)
        
        tracks = run_cotracker(frames, queries, args.online, args.start, args.end)
        
        np.save(args.output_path, tracks)
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def run_cotracker(frames, queries, online_mode, start, end):
    import torch
    
    # Your working code here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queries_tensor = torch.tensor(queries, dtype=torch.float16, device=device)[None]
    
    if start is None and end is None:
        video = np.array(frames)
    else:
        video = np.array(frames[start:end+1])
    
    video = _prepare_video_chunk(video).to(device)
    
    if online_mode:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        model = model.to(device)
        # Forward run
        print("Forward Run")
        model(video_chunk=video, is_first_step=True, queries=queries_tensor)
        for ind in range(0, video.shape[1] - model.step, model.step) :
            pred_tracks_forward, _ = model(video_chunk=video[:, ind : ind + model.step * 2])
        # Backward run
        print("Backward run")
        queries_tensor[:,:,0] = video.shape[1] - queries_tensor[:,:,0]
        video = video.flip(1)
        model(video_chunk=video, is_first_step=True, queries=queries_tensor)
        for ind in range(0, video.shape[1] - model.step, model.step) :
            pred_tracks_backward, _ = model(video_chunk=video[:, ind : ind + model.step * 2])
        # Temporal overlap
        tracks = torch.zeros_like(pred_tracks_forward)
        for ind, (timepoint, _, _) in enumerate(queries) :
            timepoint = int(timepoint)
            tracks[:, :timepoint, ind, :] = pred_tracks_backward[:, video.shape[1]-timepoint:video.shape[1], ind, :].flip(1)
            tracks[:, timepoint:, ind, :] = pred_tracks_forward[:, timepoint:, ind, :]
        tracks = tracks.cpu().numpy()[0]
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(device)
        pred_tracks, _ = model(video, queries=queries_tensor, backward_tracking=True)
        tracks = pred_tracks.cpu().numpy()[0]
    
    return tracks

def _prepare_video_chunk(window_frames):
    import torch
    frames = np.asarray(window_frames.copy())
    if frames.ndim == 3:
        frames = np.repeat(frames[..., np.newaxis], 3, axis=-1)
    video_chunk = torch.tensor(np.stack(frames)).float().permute(0, 3, 1, 2)[None]
    return video_chunk

if __name__ == '__main__':
    main()