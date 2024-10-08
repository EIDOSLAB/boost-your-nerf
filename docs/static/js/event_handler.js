document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function objectSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 5
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = './static/images/lego';
                        break;
                    case 1:
                        image.src = './static/images/mic';
                        break;
                    case 2:
                        image.src = './static/images/ship';
                        break;
                    case 3:
                        image.src = './static/images/hotdog';
                        break;
                    case 4:
                        image.src = './static/images/ficus';
                        break;
                    case 5:
                        image.src = './static/images/materials';
                        break;
                    case 6:
                        image.src = './static/images/chair';
                        break;
                    case 7:
                        image.src = './static/images/drums';
                        break;    
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '/out_0.gif';
                        break;
                    case 1:
                        image.src = image.src + '/out_1.gif';
                        break;
                    case 2:
                        image.src = image.src + '/out_2.gif';
                        break;
                    case 3:
                        image.src = image.src + '/out_3.gif';
                        break;
                    case 4:
                        image.src = image.src + '/out_4.gif';
                        break;
                    case 5:
                        image.src = image.src + '/out_5.gif';
                        break;
                }
            }

            let scene_list = document.getElementById("object-scale-recon").children;
            //console.log(scene_list)
            for (let i = 0; i < scene_list.length; i++) {
                console.log("OK")
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }