/**
 * @var videoDrawContext CanvasRenderingContext2D
 */
let videoDrawContext;
/**
 * @var perfDrawContext CanvasRenderingContext2D
 */
let perfDrawContext;
let videoScene = {
    changed: false,
    frameSize: {
        w: 0,
        h: 0,
    },
    points: [],
    mouse: {x: null, y: null},
}

let perfScene = {
    changed: false,
    times: [],
    width: 300.0,
    height: 200.0
}

function drawPerfScene() {

    const ctx = perfDrawContext;

    if (!ctx) {
        window.requestAnimationFrame(drawPerfScene);
        return;
    }

    if (!perfScene.changed) {
        window.requestAnimationFrame(drawPerfScene);
        return;
    }

    if (!perfScene.times.length) {
        window.requestAnimationFrame(drawPerfScene);
        return;
    }

    let max = 0;
    for (const t of perfScene.times) {
        if (t > max) {
            max = t;
        }
    }

    const yScale = perfScene.height / max;

    ctx.canvas.height = ctx.canvas.offsetHeight;
    ctx.canvas.width = ctx.canvas.offsetWidth;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    ctx.beginPath();
    ctx.fillStyle = 'blue';

    ctx.moveTo(0, perfScene.height);

    let x = 0;
    for (const t of perfScene.times) {
        ctx.lineTo(x, t * yScale);
        x++;
    }

    ctx.lineTo(x, perfScene.height);
    ctx.lineTo(0, perfScene.height);

    ctx.fill();
    ctx.closePath();

    window.requestAnimationFrame(drawPerfScene);
}

function drawVideoScene() {
    const ctx = videoDrawContext;

    if (!ctx) {
        window.requestAnimationFrame(drawVideoScene);
        return;
    }

    if (!videoScene.changed) {
        window.requestAnimationFrame(drawVideoScene);
        return;
    }

    ctx.canvas.height = ctx.canvas.offsetHeight;
    ctx.canvas.width = ctx.canvas.offsetWidth;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const xScale = ctx.canvas.width / videoScene.frameSize.w;
    const yScale = ctx.canvas.height / videoScene.frameSize.h;

    const mouse = videoScene.mouse;
    for (const p of videoScene.points) {
        const x = p.x * xScale;
        const y = p.y * yScale;
        const size = p.size;
        ctx.lineWidth = 1;
        const dist = Math.pow(x - mouse.x, 2) + Math.pow(y - mouse.y, 2);
        if (dist < 20*20) {
            ctx.strokeStyle = `rgba(0, 255, 0, ${size / 50})`;
        } else {
            ctx.strokeStyle = `rgba(255, 255, 0, ${size / 50})`;
        }
        if (!!p.angle) {
            ctx.moveTo(x, y);
            ctx.lineTo(x + Math.cos(p.angle) * size, y + Math.sin(p.angle) * size);
            ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.stroke();
    }

    if (!!mouse.x) {
        ctx.lineWidth = 1;
        ctx.strokeStyle = "red";
        ctx.beginPath();
        ctx.arc(mouse.x, mouse.y, 20, 0, 2 * Math.PI);
        ctx.stroke();
    }

    videoScene.changed = false;

    window.requestAnimationFrame(drawVideoScene)
}

function processFrame(text) {
    let matches;
    if (matches = text.match(/POINTS (?<pointsData>.+?)(;[A-Z]|$)/)) {
        const pointsData = matches.groups.pointsData;
        const points = pointsData.matchAll(/(?<x>[^\s]+) (?<y>[^\s]+) (?<size>[^\s]+) (?<angle>[^\s;]+)(;|$)/g);

        const videoPoints = [];

        for (const point of points) {
            let p = point.groups;
            videoPoints.push({
                x: parseFloat(p.x),
                y: parseFloat(p.y),
                size: parseFloat(p.size),
                angle: p.angle !== "-1" ? parseFloat(p.angle) / 180 * Math.PI : null,
            });
        }

        videoScene.points = videoPoints;

        videoScene.changed = true;
    }
}

function processMessage(text) {
    let matches;
    if (text.match(/^PERF/)) {
        const matches = text.match(/PERF (?<fps>[^\s]+) (?<time>[^\s]+)/).groups;

        document.getElementById('cv_fps').textContent = matches.fps;
        document.getElementById('cv_time').textContent = matches.time;

        perfScene.times.push(parseFloat(matches.time));

        if (perfScene.times.length > perfScene.width) {
            perfScene.times.shift();
        }

        perfScene.changed = true;
    } else if (matches = text.match(/^FRAME (?<w>[^\s]+) (?<h>[^\s]+) (?<data>.*)$/m)) {
        videoScene.frameSize = {
            w: parseFloat(matches.groups.w),
            h: parseFloat(matches.groups.h)
        };

        processFrame(matches.groups.data);
    }
}

document.onreadystatechange = function() {
    if (document.readyState !== 'complete') {
        return;
    }
    function connectHwCtl() {
        const hwCtl = new WebSocket('ws://' + document.location.host + '/hw_ctl');

        hwCtl.onopen = function () {
            console.log('onopen');
            document.getElementById('message-hw_ctl').textContent = 'Connected.';
        };
        hwCtl.onclose = function () {
            document.getElementById('message-hw_ctl').textContent = 'Lost connection.';
            console.log('onclose');
        };
        hwCtl.onmessage = async function (message) {
            const data = message.data;
            let text;
            if (data instanceof Blob) {
                text = await data.text();
            } else {
                text = data;
            }

            console.log("got '" + text + "'");
            const row = document.createElement('p');
            row.textContent = '>' + text;
            document.getElementById('log-hw_ctl').prepend(row);
        };
        hwCtl.onerror = function (error) {
            document.getElementById('message-hw_ctl').textContent = 'Error: ' + error;
            console.log(error);
        };
        document.getElementById('send-hw_ctl').onclick = function () {
            const input = document.getElementById('command-hw_ctl');
            const row = document.createElement('p');
            row.textContent = '<' + input.value;
            document.getElementById('log-hw_ctl').prepend(row);
            hwCtl.send(input.value);
            input.value = '';
        };

        return hwCtl;
    }
    function connectHwTm() {
        const hwTm = new WebSocket('ws://' + document.location.host + '/hw_tm');
        hwTm.onerror = function (error) {
            document.getElementById('message-hw_tm').textContent = 'Error: ' + error;
            console.log(error);
        };
        hwTm.onopen = function () {
            console.log('onopen');
            document.getElementById('message-hw_tm').textContent = 'Connected.';
        };
        hwTm.onmessage = async function (message) {
            const data = message.data;
            let text;
            if (data instanceof Blob) {
                text = await data.text();
            } else {
                text = data;
            }

            console.log("got '" + text + "'");
            const row = document.createElement('p');
            row.textContent = '>' + text;
            document.getElementById('log-hw_tm').prepend(row);
        };
        hwTm.onclose = function () {
            document.getElementById('message-hw_tm').textContent = 'Lost connection.';
            console.log('onclose');
        };

        return hwTm;
    }
    function connectCvCtl() {
        const cvCtl = new WebSocket('ws://' + document.location.host + '/cv_ctl');

        cvCtl.onopen = function () {
            console.log('onopen');
            document.getElementById('message-cv_ctl').textContent = 'Connected.';
        };
        cvCtl.onclose = function () {
            document.getElementById('message-cv_ctl').textContent = 'Lost connection.';
            console.log('onclose');
        };
        cvCtl.onmessage = async function (message) {
            const row = document.createElement('p');

            const data = message.data;
            let text;
            if (data instanceof Blob) {
                text = await data.text();
            } else {
                text = data;
            }

            console.log("got '" + text + "'");
            row.textContent = '>' + text;
            document.getElementById('log-cv_ctl').prepend(row);
        };
        cvCtl.onerror = function (error) {
            document.getElementById('message-cv_ctl').textContent = 'Error: ' + error;
            console.log(error);
        };
        document.getElementById('send-cv_ctl').onclick = function () {
            const input = document.getElementById('command-cv_ctl');
            const row = document.createElement('p');
            row.textContent = '<' + input.value;
            document.getElementById('log-cv_ctl').prepend(row);
            cvCtl.send(input.value);
            input.value = '';
        };

        return cvCtl;
    }
    function connectCvTm() {
        const cvTm = new WebSocket('ws://' + document.location.host + '/cv_tm');
        cvTm.onerror = function (error) {
            document.getElementById('message-cv_tm').textContent = 'Error: ' + error;
            console.log(error);
        };
        cvTm.onopen = function () {
            console.log('onopen');
            document.getElementById('message-cv_tm').textContent = 'Connected.';
        };
        cvTm.onmessage = async function (message) {
            const data = message.data;
            let text;
            if (data instanceof Blob) {
                text = await data.text();
            } else {
                text = data;
            }

            processMessage(text);
        };
        cvTm.onclose = function () {
            document.getElementById('message-cv_tm').textContent = 'Lost connection.';
            console.log('onclose');
        };

        return cvTm;
    }

    window.hwTm;
    window.hwCtl;

    setInterval(function () {
        // if (!window.hwTm || window.hwTm.readyState === window.hwTm.CLOSED) {
        //     window.hwTm = connectHwTm();
        //     console.log("Try to connect hwTm")
        // }
        // if (!window.hwCtl || window.hwTm.readyState === window.hwTm.CLOSED) {
        //     window.hwCtl = connectHwCtl();
        //     console.log("Try to connect hwCtl")
        // }
        if (!window.cvTm || window.cvTm.readyState === window.cvTm.CLOSED) {
            window.cvTm = connectCvTm();
            console.log("Try to connect cvTm")
        }
        if (!window.cvCtl || window.cvTm.readyState === window.cvTm.CLOSED) {
            window.cvCtl = connectCvCtl();
            console.log("Try to connect cvCtl")
        }
        initVideoCanvas();
    }, 50);

    function initVideoCanvas() {
        const canvas = document.getElementById('video_canvas');
        const video = document.getElementById('rtc_media_player');
        video.style = "position:relative;display:flex;";
        const style = canvas.style.cssText;
        canvas.style = "position:relative;display:flex;height:" + video.offsetHeight + "px;top:-" + video.offsetHeight + "px;margin-bottom:-" + video.offsetHeight + "px;width:" + video.offsetWidth + "px";

        if (canvas.style.cssText !== style) {
            videoDrawContext = canvas.getContext("2d");
            videoScene.changed = true;
        }
    }

    initVideoCanvas();

    const canvas = document.getElementById('video_canvas');
    canvas.onmousemove = function(event) {
        videoScene.mouse.x = event.offsetX;
        videoScene.mouse.y = event.offsetY;
        videoScene.changed = true;
    }

    canvas.onmouseout = function(event) {
        videoScene.mouse.x = null;
        videoScene.mouse.y = null;
        videoScene.changed = true;
    }

    perfDrawContext = document.getElementById('perf_canvas').getContext("2d")

    window.requestAnimationFrame(drawVideoScene);
    window.requestAnimationFrame(drawPerfScene);
};
