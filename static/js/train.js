$(function() {
    console.log("setting up train.js");
    var name = null;

    function ask_next() {
        $.ajax({
            url: "ask",
            type: "GET",
            success: function(data, textStatus, jqXHR) {
                var parsed = JSON.parse(data);

                if (parsed.status === "progress") {
                    var image = "data:image/jpeg;base64, " + parsed.image;
                    name = parsed.name;
                    $("#question").attr("src", image);
                } else {
                    $("#question").attr("src", "/static/img/train.png");
                    name = null;
                }
            }
        });
    }

    $("#list-subs").click(function() {
        var drop_down = $("#subsel");
        var class_list = drop_down.attr('class').split(/\s+/);

        drop_down.empty();

        function make_request(subject) {
            $.ajax({
                url: `/tell/${name}/${subject}`,
                type: "GET",
                success: function(data, textStatus, jqXHR) {
                    var parsed = JSON.parse(data);
                    if (parsed.redirect) {
                        window.location.href = parsed.redirect;
                    }
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    console.log(textStatus);
                    console.log(errorThrown);
                }
            })
        }

        if (!class_list.includes("show") && name !== null) {
            $.ajax({
                url: "subs_list",
                type: "GET",
                success: function(data, textStatus, jqXHR) {
                    var subjects = JSON.parse(data);
                    subjects.forEach(function(sub, idx) {
                        var anchor = $("<a/>", {
                            html: sub,
                        });
                        anchor.click(function() {
                            drop_down.removeClass("show");
                            make_request(sub);
                        });
                        drop_down.append(anchor);
                    });

                    function create_new_subject() {
                        var subject = prompt("Enter new subject", "name");
                        if (subject === null || subject === "") {
                            alert("Asshole!");
                        } else if (subject === "name") {
                            alert("Invalid Name!")
                        } else {
                            make_request(subject);
                        }
                        drop_down.removeClass("show");
                    }

                    var anchor = $("<a/>", {
                        html: "create new subject"
                    })
                    anchor.click(create_new_subject);
                    drop_down.append(anchor);

                    drop_down.addClass("show");
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    console.log(textStatus);
                    console.log(errorThrown);
                }
            });
        } else if (name !== null) {
            drop_down.removeClass("show");
        } else {
            alert("no image loaded");
        }
    });

    ask_next();
})