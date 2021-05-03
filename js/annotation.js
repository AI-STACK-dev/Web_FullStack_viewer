window.onload = function() {
    var easyMarker1 = new easyannotation.AnnotatorContainer(
        document.querySelector('#hallstattImg'), {
            style: 'border: 1px solid #909090',
            showClose: true,
            toolbarItems: [
                {
                    xtype: 'delete',
                    title: 'Delete selected or all elements'
                },
                new easyannotation.TextToolbarItem(),
                new easyannotation.LineToolbarItem(),
                new easyannotation.ArrowToolbarItem(),
                new easyannotation.EllipseToolbarItem(),
                new easyannotation.RectToolbarItem(),
                new easyannotation.CalloutToolbarItem(),
                new easyannotation.ImageToolbarItem(),
                new easyannotation.BlurToolbarItem(),
                {
                    itemId: 'save',
                    xtype: 'save',
                    title: 'Save changes and close annotator'
                }]
        });

    var easyMarkerRes = new easyannotation.AnnotatorContainer(document.querySelector('#hallstattImgRes'), {
        style: 'border: 1px solid #909090'
    });

    easyMarker1.show(function(dataUrl) {
        easyMarkerRes.clear();
        easyMarkerRes.loadJSON(dataUrl);
    }, easyannotation.ExportType.JSON);//JSON);

    easyMarkerRes.show(function(dataUrl) {
        easyMarker1.clear();
        easyMarker1.loadJSON(dataUrl);
    }, easyannotation.ExportType.JSON);

};

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
            var result_view = document.getElementById('annotation_table');
            console.log('통신 데이터 전송 성공! ^^');
            console.log(result_view);
            var data = JSON.parse(this.response);
            var template = '';
            var photos = data.results;
            console.log(name);
            var temp_name = name.split('.');
            var tempstr = temp_name[0];
            console.log(temp_name[0]);
            var idx = photos.findIndex(i => i.name== tempstr );
            console.log(idx);
            template = [
                '<tr><td>' +
                '<img style="margin-top: 50px; width:800px;height:800px;" id="hallstattImg" src="./img/'+ name +'"/>' +
                '</td><td>' +
                '<img style="margin-top: 50px; width:800px;height:800px;" id="hallstattImgRes" src="./img/'+ name +'"/>' +
                '</td></tr>'
            ].join('');
        } else {
            console.log('통신 데이터 전송 실패');
        }
        console.log(template);
        result_view.innerHTML = template;
    }

})(this, this.XMLHttpRequest || this.ActiveXObject('Microsoft.XMLHTTP'));
