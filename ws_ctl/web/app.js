/**
 * @var videoDrawContext CanvasRenderingContext2D
 */
let videoDrawContext;
/**
 * @var videoDrawHiddenContext CanvasRenderingContext2D
 */
let videoDrawHiddenContext;
/**
 * @var videoDrawHiddenCanvas HTMLCanvasElement
 */
let videoDrawHiddenCanvas = null;
/**
 * @var perfDrawContext CanvasRenderingContext2D
 */
let perfDrawContext;
let videoScene = {
    changed: false,
    frameSize: {
        w: 300,
        h: 200,
    },
    points: [],
    mouse: {x: null, y: null},
    image: null,
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
    const ctx = videoDrawHiddenContext;

    if (!ctx) {
        window.requestAnimationFrame(drawVideoScene);
        return;
    }

    if (ctx.isContextLost()) {
        console.log("ctx.isContextLost()");
        return;
    }

    if (!videoScene.changed) {
        window.requestAnimationFrame(drawVideoScene);
        return;
    }

    const xScale = ctx.canvas.width / videoScene.frameSize.w;
    const yScale = ctx.canvas.height / videoScene.frameSize.h;

    if (videoScene.image !== null) {
        ctx.drawImage(videoScene.image, 0, 0);
    }

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

    videoDrawContext.drawImage(videoDrawHiddenCanvas, 0, 0);

    window.requestAnimationFrame(drawVideoScene)
}

function processFrame(text) {
    let matches;
    if ((matches = text.match(/POINTS (?<pointsData>.+?)(;[A-Z]|$)/)) || (matches = text.match(/POINTS/))) {
        const pointsData = matches.groups && matches.groups.pointsData;

        if (!pointsData) {
            videoScene.points = [];
            videoScene.changed = true;
            return;
        }

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

function processConfig(text) {
    let matches;
    if (matches = text.match(/denoiseLevel\s(?<value>\S+)/)) {
        const level = matches.groups.value;

        const denoiseLevelInput = document.getElementById('denoiseLevel');
        const denoiseLevelView = document.getElementById('denoiseLevelView');

        denoiseLevelInput.value = level;
        denoiseLevelView.textContent = level;
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
    } else if (matches = text.match(/^CONFIG (?<data>.*)$/m)) {
        processConfig(matches.groups.data);
    }
}

function processStreaming(text) {
    console.log("Frame received " + text.length);
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
            const data = message.data;
            let text;
            if (data instanceof Blob) {
                text = await data.text();
            } else {
                text = data;
            }

            processMessage(text);
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

    function connectCvS() {
        const cvS = new WebSocket('ws://' + document.location.host + '/cv_s');
        cvS.binaryType = 'blob';
        cvS.onerror = function (error) {
            document.getElementById('message-cv_s').textContent = 'Error: ' + error;
            console.log(error);
        };
        cvS.onopen = function () {
            console.log('onopen');
            document.getElementById('message-cv_s').textContent = 'Connected.';
        };
        cvS.onmessage = async function (message) {
            const data = message.data;
            if (data instanceof Blob) {
                let image = null;
                try {
                    image = await createImageBitmap(data);
                } catch (e) {
                    if (videoScene.image === null) {
                        console.log(e);
                    }
                }
                if (image !== null) {
                    videoScene.image = image;
                    videoScene.changed = true;
                }
            } else {
                throw "!data instanceof Blob";
            }
        };
        cvS.onclose = function () {
            document.getElementById('message-cv_s').textContent = 'Lost connection.';
            console.log('onclose');
        };

        return cvS;
    }

    window.hwTm;
    window.hwCtl;

    setInterval(function () {
        // if (!window.hwTm || window.hwTm.readyState === window.hwTm.CLOSED) {
        //     window.hwTm = connectHwTm();
        //     console.log("Try to connect hwTm")
        // }
        // if (!window.hwCtl || window.hwCtl.readyState === window.hwCtl.CLOSED) {
        //     window.hwCtl = connectHwCtl();
        //     console.log("Try to connect hwCtl")
        // }
        if (!window.cvTm || window.cvTm.readyState === window.cvTm.CLOSED) {
            window.cvTm = connectCvTm();
            console.log("Try to connect cvTm")
        }
        if (!window.cvCtl || window.cvCtl.readyState === window.cvCtl.CLOSED) {
            window.cvCtl = connectCvCtl();
            console.log("Try to connect cvCtl")
        }
        if (!window.cvS || window.cvS.readyState === window.cvS.CLOSED) {
            window.cvS = connectCvS();
            console.log("Try to connect cvS")
        }
        initVideoCanvas();
    }, 500);

    function initVideoCanvas() {
        const canvas = document.getElementById('video_canvas');
        const hiddenCanvas = videoDrawHiddenCanvas === null
            ? document.createElement('canvas')
            : videoDrawHiddenCanvas;

        if (videoScene.image !== null) {
            videoScene.frameSize = {
                w: videoScene.image.width,
                h: videoScene.image.height,
            }
        }

        const frameSize = videoScene.frameSize;

        canvas.width = frameSize.w;
        canvas.height = frameSize.h;
        hiddenCanvas.width = frameSize.w;
        hiddenCanvas.height = frameSize.h;
        videoDrawContext = canvas.getContext("2d", { alpha: false });
        videoDrawHiddenContext = hiddenCanvas.getContext("2d", { alpha: false });
        videoScene.changed = true;

        videoDrawHiddenCanvas = hiddenCanvas;
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

    const denoiseLevelInput = document.getElementById('denoiseLevel');
    const denoiseLevelView = document.getElementById('denoiseLevelView');
    denoiseLevelInput.onmousemove = function (event) {
        const level = parseFloat(denoiseLevelInput.value);
        if (denoiseLevelInput.value !== denoiseLevelView.textContent) {
            denoiseLevelView.textContent = level;
            window.cvCtl.send("SETCONFIG denoiseLevel " + level);
        }
    }
};
