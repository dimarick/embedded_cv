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
        hwCtl.onmessage = function (message) {
            console.log("got '" + message.data + "'");
            const row = document.createElement('p');
            row.textContent = '>' + message.data;
            document.getElementById('log-hw_ctl').appendChild(row);
        };
        hwCtl.onerror = function (error) {
            document.getElementById('message-hw_ctl').textContent = 'Error: ' + error;
            console.log(error);
        };
        document.getElementById('send-hw_ctl').onclick = function () {
            const input = document.getElementById('command-hw_ctl');
            const row = document.createElement('p');
            row.textContent = '<' + input.value;
            document.getElementById('log-hw_ctl').appendChild(row);
            cvCtl.send(input.value);
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
            const text = await message.data.text();
            console.log("got '" + text + "'");
            const row = document.createElement('p');
            row.textContent = '>' + text;
            document.getElementById('log-hw_tm').appendChild(row);
        };
        hwTm.onclose = function () {
            document.getElementById('message-hw_tm').textContent = 'Lost connection.';
            console.log('onclose');
        };

        return hwTm;
    }

    let hwTm;
    let hwCtl;

    setInterval(function () {
        if (!hwTm || hwTm.CLOSED) {
            hwTm = connectHwTm();
        }
        if (!hwCtl || hwCtl.CLOSED) {
            hwCtl = connectHwCtl();
        }
    }, 50);

    // cvCtl = new WebSocket('ws://' + document.location.host + '/cv_ctl');
    // cvTm = new WebSocket('ws://' + document.location.host + '/cv_tm');
    //
    // cvCtl.onopen = function() {
    //     console.log('onopen');
    //     document.getElementById('message-cv_ctl').textContent = 'Connected.';
    // };
    // cvTm.onopen = function() {
    //     console.log('onopen');
    //     document.getElementById('message-cv_tm').textContent = 'Connected.';
    // };
    // cvCtl.onclose = function() {
    //     document.getElementById('message-cv_ctl').textContent = 'Lost connection.';
    //     console.log('onclose');
    // };
    // cvCtl.onmessage = function(message) {
    //     console.log("got '" + message.data + "'");
    //     const row = document.createElement('p');
    //     row.textContent = '>' + message.data;
    //     document.getElementById('log-cv_ctl').appendChild(row);
    // };
    // cvTm.onmessage = function(message) {
    //     console.log("got '" + message.data + "'");
    //     const row = document.createElement('p');
    //     row.textContent = '>' + message.data;
    //     document.getElementById('log-cv_tm').appendChild(row);
    // };
    // cvCtl.onerror = function(error) {
    //     document.getElementById('message-cv_ctl').textContent = 'Error: ' + error;
    //     console.log(error);
    // };
    // cvTm.onerror = function(error) {
    //     document.getElementById('message-cv_tm').textContent = 'Error: ' + error;
    //     console.log(error);
    // };
    // document.getElementById('send-cv_ctl').onclick = function() {
    //     const input = document.getElementById('command-cv_ctl');
    //     const row = document.createElement('p');
    //     row.textContent = '<' + input.value;
    //     document.getElementById('log-cv_ctl').appendChild(row);
    //     cvCtl.send(input.value);
    //     input.value = '';
    // };

};

set = function(value) {
	$('#count').val(value)
}
