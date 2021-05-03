function getUrlValue(key) {
    //console.log(key);
    var valueObject = {}, hash, value;
    var hashes = window.location.href.slice(window.location.href.indexOf('?'));
    //console.log(hashes);
    hash = hashes.split('=');
    //console.log(hash);
    valueObject[hash] = hash[1];
    //console.log(valueObject[hash]);
    num = valueObject[hash];
    if (key) {

        if (valueObject[key]) {
            return valueObject[key];
        }

        return "";
    }
    console.log(valueObject);
    return valueObject[hash];
}
var name = getUrlValue();


(function (global, XHR) {
    'use strict';

    var xhr = new XHR;
    xhr.open("GET", "./data/gallery.json");
    xhr.send();
    xhr.onreadystatechange = function () {
        if (this.status === 200 && this.readyState === 4) {
            var result_view = document.querySelector('.ajax-results');
            console.log('통신 데이터 전송 성공! ^^');
            console.log(result_view);
            var data = JSON.parse(this.response);
            var template = '';
            var photos = data.results;
            var AI_results = data.AI;
            console.log(name);
            var temp_name = name.split('.');
            var tempstr = temp_name[0];
            console.log(temp_name[0]);
            var idx = photos.findIndex(i => i.name== tempstr );
            console.log(idx);

            template = [
                '<div class = "zoom">',
                '<div class = "row">',
                '<div class = "col-lg-6">',
                '<img class="photo-img"  id="ex5" src="' + photos[idx].image + '"   width ="'+ 800 +'" height="'+800+'"  alt="' + photos[idx].alt + '">',
                '</div>',
                '<div class= "col-lg-6">',
                '<img class="photo-img"  id="ex5" src="' + AI_results[0].image + '"   width ="'+ 800 +'" height="'+800+'"  alt="' + AI_results[0].alt + '">',
                '</div>',
                '</div>',
                '</div>',
            ].join('');
        } else {
            console.log('통신 데이터 전송 실패');
        }
        console.log(template);
        result_view.innerHTML = template;
    }

})(this, this.XMLHttpRequest || this.ActiveXObject('Microsoft.XMLHTTP'));
