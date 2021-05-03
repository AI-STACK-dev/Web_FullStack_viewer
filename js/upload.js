var ajaxCall;

$('.upload-btn').on('click', function (){
    $('#upload-input').click();
    $('.progress-bar').text('0%');
    $('.progress-bar').width('0%');
});
$('#upload-input').on('change', function(){
    var files = $(this).get(0).files;
    if (files.length > 0){
        // create a FormData object which will be sent as the data payload in the
        // AJAX request
        var formData = new FormData();
        // loop through all the selected files and add them to the formData object
        for (var i = 0; i < files.length; i++) {
            var file = files[i];
            // add the files to formData object for the data payload
            formData.append('uploads[]', file, file.name);
        }
       ajaxCall = $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            beforeSend : function(xhr, opts) {
                // when validation is false
                if (false) {
                    xhr.abort();
                }
            },
            xhr: function() {
                // create an XMLHttpRequest
                var xhr = new XMLHttpRequest();
                console.log(xhr);
                // listen to the 'progress' event
                xhr.upload.addEventListener('progress', function(evt) {
                    if (evt.lengthComputable) {
                        // calculate the percentage of upload completed
                        console.log(evt.loaded+'/'+ evt.total);
                        var percentComplete = evt.loaded / evt.total;
                        percentComplete = parseInt(percentComplete * 100);

                        // update the Bootstrap progress bar with the new percentage
                        $('.progress-bar').text(percentComplete + '%');
                        $('.progress-bar').width(percentComplete + '%');
                        // once the upload reaches 100%, set the progress bar text to done
                        if (percentComplete === 100) {
                            $('.progress-bar').html('Done');
                        }
                    }
                }, false);
                return xhr;
            },
            success: function(data){
                console.log('upload successful!\n' + data);
            },
            error : function(request,status,error) {
                alert("code:"+request.status+'\n'+"message:"+request.responseText+'\n'+"error:"+error);
            }
        });
        $(document).on('click','.stopupload', function(e){
            ajaxCall.abort();
            console.log("Canceled");
        });
    }
});
