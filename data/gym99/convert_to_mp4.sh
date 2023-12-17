task(){
   /nvme/wangyiqin/ffmpeg-git-20220910-amd64-static/ffmpeg -thread_queue_size 512 -framerate 12 -i "$1/img_%05d.jpg" -start_number 1 -c:v libx264 "$1.mp4";
}

N=32
(
for f in *; do
   ((i=i%N)); ((i++==0)) && wait
   task "$f" &
done
)
