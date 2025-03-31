#!/bin/bash
# NB: manually ran videos with spaces in their names

video_clips_dir=/data/commcon/stimuli/commcon_videos_all
frames_base_dir=/data/commcon/stimuli/frames_by_video
output_base_dir=/data/commcon/stimuli/output_by_video


for filename in $video_clips_dir/*.mp4; do
	video_name=$(basename $filename)
	video_name_noext=${video_name%%.*}

	echo ---working on $video_name_noext---

	cur_frames_dir=$frames_base_dir/"$video_name_noext"_frames

	if [ ! -d "$cur_frames_dir" ]; then
	  echo "$cur_frames_dir does not exist - creating."
	  mkdir $cur_frames_dir
	fi

	cur_output_dir=$output_base_dir/"$video_name_noext"_output

	if [ ! -d "$cur_output_dir" ]; then
	  echo "$cur_output_dir does not exist - creating."
	  mkdir $cur_output_dir
	fi

	if [ -n "$(ls -A $cur_output_dir)" ]; then
	    echo "$cur_output_dir already contains files - not rerunning imgtofmri"
	else
	    img2fmri --input $cur_frames_dir --output $cur_output_dir
	fi

done
