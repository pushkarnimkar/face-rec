var files = {}, tmp, canvas;


function base64_to_jpeg(image) {
    var blobBin = atob(image.split(',')[1]);
    var array = new Uint8Array(blobBin.length);
    for(var i = 0; i < blobBin.length; i++) {
        array[i] = blobBin.charCodeAt(i);
    }
    var file = new Blob([array], {type: "image/jpeg"});
    return file;
}


$(function() {
    console.log("setting up feed.js");

    $("#upload").click(function() {
        console.log("clicked upload");
        $("#image").click();
        $("#status").text("Preview Uploaded Image");
    })

    $("#image").on("change", function(event) {
        files["image"] = event.target.files[0];

        var reader = new FileReader();
        reader.onload = function(e) {
            console.log(e);
            $("#tagged").attr("src", e.target.result);
        }
        reader.readAsDataURL(files["image"]);
    });

    $("#submit").click(function() {
        if (files["image"] === undefined) {
            alert("no image loaded");
        } else {
            var data = new FormData();
            data.append("image", files["image"]);

            $.ajax({
                url: "feed",
                type: "POST",
                data: data,
                processData: false,
                contentType: false,
                success: function(data, textStatus, jqXHR) {
                    var parsed = JSON.parse(data)
                    var image = "data:image/jpeg;base64, " + parsed.image;

                    $("#tagged").attr("src", image);
                    $("#status").text(parsed.status);
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    console.log(textStatus);
                    console.log(errorThrown);
                }
            });
        }
    });

    $("#webcam").click(function() {
        var video  = $("<video id=\"video\" autoplay></video>")[0];
        $("#tagged").replaceWith(video);

        navigator.mediaDevices.getUserMedia({video: true})
            .then(function(stream) {
                video.srcObject = stream;
                $("#video").click(function() {
                    var canvas = $("<canvas style=\"display:none;\"></canvas>")[0];
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;

                    canvas.getContext("2d").drawImage(video, 0, 0);
                    var image = canvas.toDataURL("image/jpeg");

                    console.log(image);
                    var tagged = $("<img id=\"tagged\" height=\"480px\" width=\"640px\">")[0];
                    tagged.src = image;

                    files["image"] = base64_to_jpeg(image);

                    video.srcObject.getTracks()[0].stop();
                    video.replaceWith(tagged);
                });
            });
    });

    $("#acquire").click(function() {
        $.ajax({
            url: "acquire",
            type: "GET",
            processData: false,
            contentType: false,
            success: function(data, textStatus, jqXHR) {
                var parsed = JSON.parse(data)
                if (parsed.hasOwnProperty("image")) {
                    var image = "data:image/jpeg;base64, " + parsed.image;
                    files["image"] = base64_to_jpeg(image);
                    $("#tagged").attr("src", image);
                }
                $("#status").text(parsed.status);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.log(textStatus);
                console.log(errorThrown);
            }
        });
    });
});
