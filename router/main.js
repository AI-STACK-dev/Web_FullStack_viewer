var express = require('express');
var app = express();
var fs= require('fs');
var path = require('path');
var multer = require('multer'); // multer모듈 적용 (for 파일업로드)
var multiparty = require('multiparty');
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'public/img/') // cb 콜백함수를 통해 전송된 파일 저장 디렉토리 설정
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname) // cb 콜백함수를 통해 전송된 파일 이름 설정
    }
});
var upload = multer({ storage: storage })
module.exports = function(app)
{

    //기본 root 홈페이지를 불러오는 부분입니다.
    app.get('/',function(req,res){
        res.render('default.html');
    });
    app.get('/biomedical_Login.html',function(req,res){
      res.render("biomedical_Login.html");
    })
    app.get('/python_test.html',function(req,res){
      res.render("python_test.html");
    })
    app.get('/biomedical_register.html',function (req,res){
        res.render("biomedical_register.html");
    })
    app.get('/annotation.html',function(req,res){
        //var name = req.params.id;
        //console.log(name);
        //var temp_name = name.split('.');
        //var file_name = temp_name[0];
        res.render("annotation.html");
    });
    //viewer button을 html에서 클릭 했을시 viwer창을 띄워 주는 부분입니다.
    app.get('/test.html',function(req,res){
        res.render('test.html');
    });
    app.get('/viewer.html',function(req,res){
        res.render('viewer.html');
    });
    app.get('/results.html',function(req,res){
      res.render('results.html');
    })
    app.get('/AI/:id',function(req,res){

        var filename = req.params.id;
        var Pythonshell = require("python-shell");
        var fs = require("fs");

// nodejs에서 이미지파일 받기
        var data = fs.readFileSync("./public/img/"+filename);
        var base64 = data.toString("base64");
        //console.log(base64);
        pyshell = new Pythonshell.PythonShell('./scripts1.py', {
            pythonPath: '/Users/janghyeonjun/opt/anaconda3/bin/python'
        });

//base64로 인코딩한 문자열을 전송
        pyshell.send(base64);

        pyshell.on("message", rebase64 => {
            var err = null;

            //base64 문자열을  Buffer(bytes)타입으로 디코딩
           result = Buffer.from(rebase64,"base64");
           console.log(result);
            fs.writeFileSync("./public/img/result.png", result);
            var filename = './public/img/result.png';
            fs.readFile(filename,function(err,data){
                console.log('readsuccess');
                res.redirect('/results.html?name='+req.params.id);
                //res.writeHead(200, { "Context-Type": "image/png" });//보낼 헤더를 만듬
               // res.write(data);   //본문을 만들고
                //res.end();  //클라
            });
        });

        pyshell.end(err => {
            if (err) {
                console.log(err);
            }
        });
    });

    app.get('/python',function(req,res){
        //res.send('성공');

        var Pythonshell = require("python-shell");
        var fs = require("fs");

// nodejs에서 이미지파일 받기
        var data = fs.readFileSync("./public/img/19.jpg");
        var base64 = data.toString("base64");

        pyshell = new Pythonshell.PythonShell('./scripts1.py', {
            pythonPath: '/Library/Developer/CommandLineTools/usr/bin/python3'
        });

//base64로 인코딩한 문자열을 전송
        pyshell.send(base64);

        pyshell.on("message", rebase64 => {
            var err = null;

            //base64 문자열을  Buffer(bytes)타입으로 디코딩
            result = Buffer.from(rebase64,"base64");
            console.log(result);
            fs.writeFileSync("./public/img/result.jpeg", result);
            var filename = './public/img/result.jpeg';
            fs.readFile(filename,function(err,data){
                res.writeHead(200, { "Context-Type": "image/jpg" });//보낼 헤더를 만듬
                res.write(data);   //본문을 만들고
                res.end();  //클라
            });
        });

        pyshell.end(err => {
            if (err) {
                console.log(err);
            }
        });



        /*
        const {PythonShell} = require('python-shell');
        let options = {
            mode: 'text',
            pythonPath: '/Library/Developer/CommandLineTools/usr/bin/python3',
            pythonOptions: ['-u'],
            args: ['value1', 'value2','value3']
        };

        let pyshell = new PythonShell('./python1.py',options);
        console.log(pyshell);
        pyshell.send('hello');
        pyshell.send('world');

        pyshell.on('msg',(msg) => {
            console.log(msg);
            res.send(msg);
        })

        pyshell.end((err,code,signal) =>{
            if(err) throw err;
            console.log('The exit code was: ' + code);
            console.log('The exit signal was:' + signal);
            console.log('finished');
        })
    */

        /*2단계
        PythonShell.run('./python.py', options, (err, msg) => {
            if (err) throw err;
            console.log('results: %j', msg);
            res.send(msg);
        })
            */


        //1단
        //PythonShell.runString('x=1+1; print(x); y = 2+2; print(y)' , null, (err, msg) => {
          //  console.log('err :', err)
           // console.log('msg :', msg)
           // res.send(msg);
        //})
    })
    app.get('/logout',function(req,res){
        console.log(req);
        req.logout();
        delete req.session;
        res.redirect('/');
    });
    //원래는 delete 팝업창을 만들어놨지만 지금은 confirm창으로 동작하게 했습니다.
    app.get('/delete.html',function(req,res){
       res.render('delete.html');
    });
    //description update는 팝업창을 구현 하려 했지만 페이지로 구현했습니다.
    app.get('/update.html',function (req,res){
        res.render('update.html');
    });
    //web 서비스적으로 uploader를 할수 있게 uploader페이지를 불러주는 부분입니다.
    app.get('/uploader.html',function(req,res){
        res.render('uploader.html');
    });
    app.get('/Login.html',function (req,res){
      res.render('Login.html');
    });
    //웹상에서는 ajax로 통신을하지만 mfc를 이용한 통신을 해야하므로 post요청을 받는 부분입니다.
    app.post('/upload',upload.any(),function (req,res){
        console.log(req.files[0]);
        res.send('Uploaded 업로드를 성공하였습니다');
        //mfc에게 업로드요청이 되었다고 response 해주는 부분입니다.

        var tempstr =  req.files[0].originalname;
        console.log(tempstr);
        var temp_name = tempstr.split('.');
        var name = tempstr;
        console.log(name);
        var element = {
            name: temp_name[0],
            image: './img/'+tempstr,
            alt: "pathology Image"
        };
        //파일을 다운로드를 받은뒤 웹서비스상에는 JSON 파일을 바꿔야 하므로 새로 추가된 객체를 만드는 부분입니다.

        var str_element = element.toString();
        console.log(typeof (str_element));
        console.log('element='+typeof(element));
        const fs = require('fs');
        var storage = {};

        console.log('?');
        //여기서 부터는 서버에 존재하는 json 파일을 오픈한뒤 위에서 만든 JSON객체를 저장하는 부분입니다.
        fs.readFile('./public/data/gallery.json',function (err,data){
            if(data){
                console.log("Read JSON file:" + data);
                console.log(typeof(data));
                var data = data.toString();
                storage = JSON.parse(data.trim());
                console.log(typeof(storage));
                var photos = storage.results;
                console.log('photos =' + typeof(photos));
                console.log(photos);
                console.log(element);
                photos[photos.length] = element;
                console.log(photos);
                var photosArray = JSON.stringify(photos);
                var temp = '{"results":'+ photosArray +'}';
                console.log(temp);


                fs.writeFileSync('./public/data/gallery.json',temp);
                //객체를 추가를 한뒤 JSON을 다시 문자열로변경하여 JSON파일에 저장하는 부분입니다.
                console.log('completed');
                //서버에 JSON까지 전부변경을 확인하려는 의미에서 넣은 출력문입니다.

            }
        });
    });
    //다음은 다운로드 get요청이 들어왔을때 처리해주는 부분입니다.
    app.get('/download/:id',function(req,res){
       var path = './public';
       var filename = req.params.id;
       console.log(filename);
       let filepath = path + "/img/" + filename;
       res.download(filepath);
       //위에서 문자열로 file path -> 다운로드 받아얄 파일의 경로를 확인하고 바로 위에 코드에서 다운을 할수 있게 됩니다.
    });
    app.get('/update/:id',function(req,res){
        //업데이트는 웹서비스이므로 업데이트 완료된 뒤 바로 홈으로 리다이렉트 해주는 부분입니다.
        res.redirect('/');

        var temp_filename= req.params.id;
        var temp_name = temp_filename.split('&');
        var name = temp_name[0];
        var description = temp_name[1];

        //update된 부분에 대해서 parsing을 해주는 부분입니다.

        //get을 통해 이미 id를 받아와서 JSON 파일에서의 일치하는 id를 검색한뒤 그 부분만 변경하는걸로 진행했습니다
        fs.readFile('./public/data/gallery.json',function(err,data) {
            if(data){
                var storage = {};
                var data = data.toString();
                storage = JSON.parse(data.trim());
                var photos = storage.results;
                var idx = photos.findIndex(i => i.name== name );
                //index를 찾은 뒤
                photos[idx].alt = description;
                //index의 설명을 변경하는 부분입니다.
                var photosArray = JSON.stringify(photos);
                var temp = '{"results":'+ photosArray +'}';
                fs.writeFileSync('./public/data/gallery.json',temp);
                console.log('completed');
            }

        });
    })

    app.get('/delete/:id',function(req,res){
        var temp_filename= req.params.id;
        var temp_name = temp_filename.split('.');
        var name = temp_name[0];

        fs.readFile('./public/data/gallery.json',function(err,data){
            if(data){
                var storage = {};
                var data = data.toString();
                storage = JSON.parse(data.trim());
                var photos = storage.results;
                var idx = photos.findIndex(i => i.name== name );
                //전반적으로 parsing을 한뒤 index를 찾는 과정은 동일합니다.
                photos.splice(idx,1);
                //JSON객체를 자르는 부분입니다.
                var photosArray = JSON.stringify(photos);
                var temp = '{"results":'+ photosArray +'}';
                fs.writeFileSync('./public/data/gallery.json',temp);
                console.log('completed');
            }
        });

        var path = './public';
        var filename = req.params.id;
        let filepath = path + "/img/" + filename
        //실질적으로 파일을 제거하는 부분입니다.
        fs.unlink(filepath,function(){
            res.redirect('/');
            //제거해준뒤 root page로 돌아가데 만들어주는 부분입니다.
            res.end();
        });
    });
}
