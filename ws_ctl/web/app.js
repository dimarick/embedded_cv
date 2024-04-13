var ws;

document.onreadystatechange = function() {
    if (document.readyState !== 'complete') {
        return;
    }
    ws = new WebSocket('ws://' + document.location.host + '/ws');
    ws.onopen = function() {
        console.log('onopen');
        document.getElementById('message').textContent = 'Connected.';
    };
    ws.onclose = function() {
        document.getElementById('message').textContent = 'Lost connection.';
        console.log('onclose');
    };
    ws.onmessage = function(message) {
        console.log("got '" + message.data + "'");
        document.getElementById('message').textContent = message.data;
        document.getElementById('count').value = message.data;
    };
    ws.onerror = function(error) {
        console.log('onerror ' + error);
        console.log(error);
    };
    document.getElementById('count').onclick = function() {
    	ws.send(document.getElementById('count').value);
    };
    document.getElementById('close').onclick = function() {
      ws.send('close');
    };
    document.getElementById('die').onclick = function() {
      ws.send('die');
    };
};

set = function(value) {
	$('#count').val(value)
}
