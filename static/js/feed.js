var files = {}, tmp;

$(function() {
    console.log("setting up feed.js");
    $("#image").on("change", function(event) {
        files["image"] = event.target.files[0];

        var reader = new FileReader();
        reader.onload = function(e) {
            console.log(e);
            $("#tagged").attr("src", e.target.result);
        }
        reader.readAsDataURL(files["image"]);
    });

    $("#image_upload").on("submit", function(event) {
        event.stopPropagation();
        event.preventDefault();

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
    });
});
