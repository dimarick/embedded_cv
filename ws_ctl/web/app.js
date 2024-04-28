var ws;

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

            console.log("got '" + text + "'");
            const row = document.createElement('p');
            row.textContent = '>' + text;
            document.getElementById('log-cv_tm').prepend(row);
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
        if (!window.hwTm || window.hwTm.readyState === window.hwTm.CLOSED) {
            window.hwTm = connectHwTm();
            console.log("Try to connect hwTm")
        }
        if (!window.hwCtl || window.hwTm.readyState === window.hwTm.CLOSED) {
            window.hwCtl = connectHwCtl();
            console.log("Try to connect hwCtl")
        }
        if (!window.cvTm || window.cvTm.readyState === window.cvTm.CLOSED) {
            window.cvTm = connectCvTm();
            console.log("Try to connect cvTm")
        }
        if (!window.cvCtl || window.cvTm.readyState === window.cvTm.CLOSED) {
            window.cvCtl = connectCvCtl();
            console.log("Try to connect cvCtl")
        }
    }, 50);
};

set = function(value) {
	$('#count').val(value)
}
