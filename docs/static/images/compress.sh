#!/bin/bash
scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")


for scene in "${scenes[@]}"; do
    for i in {0..4}; do
        # Crea una palette per ciascun file
        echo $scene
        ffmpeg -y -i "$scene/out_$i.gif" -vf "fps=10,scale=iw/2:ih/2,palettegen" "$scene/palette_$i.png"

        # Usa la palette per ridurre i colori e comprimere ciascun file
        ffmpeg -y -i "$scene/out_$i.gif" -i "$scene/palette_$i.png" -lavfi "fps=10,scale=iw/2:ih/2,paletteuse" -compression_level 10 "$scene/out_cmp_$i.gif"
    done
done