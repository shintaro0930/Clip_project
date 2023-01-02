// 画像を切り替える関数
function changeImage() {
    idx++;
    if (idx > 3) {  
        idx = 1;
    }

    // img要素のsrcに画像ファイル名を設定する   
    img.src = "/project/used_pictures/" + idx + ".jpg";
}

// idタグを取得
let img = document.getElementById("ouptut");
