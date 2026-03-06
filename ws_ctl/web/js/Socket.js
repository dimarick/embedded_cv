export default class Socket {
    static #bytesSent0 = 0;
    static #bytesReceived0 = 0;
    static #bytesSent = 0;
    static #bytesReceived = 0;
    static #bytesSentPrev = 0;
    static #bytesReceivedPrev = 0;
    static #statTs = 0;
    static #bpsSent = 0;
    static #bpsReceived = 0;
    static #bpsInterval = 0;

    #url;
    onmessage;
    #ws;
    #connected = false;
    #connecting = false;
    #receiveMessageQueue = [];
    #sendMessageQueue = [];
    #waitingResolvers = [];
    #onmessage = null;

    constructor(url, onmessage) {
        this.#onmessage = onmessage || (() => {});
        this.#url = url;
        this.connect();
        setInterval(() => {
            this.connect();
        }, 2000);

        if (Socket.#bpsInterval === 0) {
            Socket.#statTs = performance.now();
            Socket.#bpsInterval = setInterval(() => Socket.#sendStat(), 1000);
        }
    }

    connect() {
        if (this.#connecting || this.#connected) {
            return;
        }
        this.#connecting = true;
        this.#ws = new WebSocket('ws://' + document.location.host + this.#url);

        this.#ws.onopen = () => {
            console.log('onopen');
            document.getElementById('connection-status').textContent = 'Connected.';
            this.#connected = true;
            this.#connecting = false;

            for (const message of this.#sendMessageQueue) {
                this.#ws.send(message);
            }
        };
        this.#ws.onclose = () => {
            document.getElementById('connection-status').textContent = 'Lost connection.';
            console.log('onclose');
            this.#connected = false;
            this.#connecting = false;
        };
        this.#ws.onmessage = async (message) => {
            const data = message.data;

            const dataSize = data instanceof Blob ? data.size : data.length;
            Socket.#bytesReceived += dataSize;
            Socket.#bytesReceived0 += dataSize;

            if (typeof data === "string" && data.indexOf("ERROR ") === 0) {
                document.getElementById('connection-status').textContent = 'Error: ' + data;
                console.log(data);
                window.logger.error("Сбой подключения к " + this.#ws.url + " причина " + data);
                this.#connecting = false;
                this.#ws.close();
                return;
            }

            // Если есть ожидающие вызовы getNextMessage, резолвим первый из них
            if (this.#waitingResolvers.length > 0) {
                const resolve = this.#waitingResolvers.shift();
                resolve(data);
            } else {
                // Иначе сохраняем в очередь
                this.#receiveMessageQueue.push(data);
            }

            if (!!this.#onmessage) {
                this.#onmessage(data);
            }
        };
        this.#ws.onerror = (error) => {
            document.getElementById('connection-status').textContent = 'Error: ' + error;
            console.log(error);
            window.logger.error("Не удалось подключиться к " + this.#ws.url + " причина " + error.reason);
            this.#connecting = false;
        };
    }

    disconnect(code, reason) {
        this.#ws.close(code, reason);
    }

    connected() {
        return this.connected;
    }

    async getNextMessage() {
        if (this.#receiveMessageQueue.length > 0) {
            return this.#receiveMessageQueue.shift();
        }

        // Иначе создаём промис и сохраняем его resolve в очередь ожидания
        return new Promise((resolve) => {
            this.#waitingResolvers.push(resolve);
        });
    }

    sendMessage(data) {
        this.connect();
        if (!this.#connected) {
            this.#sendMessageQueue.push(data);
        } else {
            this.#ws.send(data);
        }

        Socket.#bytesSent += data.length;
        Socket.#bytesSent0 += data.length;
    }

    static #sendStat() {
        const now = performance.now();
        const sent = Socket.#bytesSent;
        const recv = Socket.#bytesReceived;

        const interval = now - Socket.#statTs;
        Socket.#bpsSent = 1000 * (sent - Socket.#bytesSentPrev) / interval;
        Socket.#bpsReceived = 1000 * (recv - Socket.#bytesReceivedPrev) / interval;
        Socket.#bytesSentPrev = sent;
        Socket.#bytesReceivedPrev = recv;
        Socket.#statTs = now;

        if (Socket.#bytesSent > 1e8) {
            Socket.#bytesSent -= Socket.#bytesSentPrev;
            Socket.#bytesSentPrev -= Socket.#bytesSentPrev;
        }

        if (Socket.#bytesReceived > 1e8) {
            Socket.#bytesReceived -= Socket.#bytesReceivedPrev;
            Socket.#bytesReceivedPrev -= Socket.#bytesReceivedPrev;
        }

        window.telemetryStatus.setStatusAll('network', [
            'bytes.sent.raw', Socket.#bytesSent0,
            'bytes.sent.value', Math.round(Socket.#bytesSent0 / 1e5) / 1e1,
            'bytes.sent.unit', 'MB',
            'bytes.received.raw', Socket.#bytesReceived0,
            'bytes.received.value', Math.round(Socket.#bytesReceived0 / 1e5) / 1e1,
            'bytes.received.unit', 'MB',
            'bps.sent.raw', Socket.#bpsSent,
            'bps.sent.value', Math.round(Socket.#bpsSent / 1e3),
            'bps.sent.unit', 'kB/s',
            'bps.received.raw', Socket.#bpsReceived,
            'bps.received.value', Math.round(Socket.#bpsReceived / 1e3),
            'bps.received.unit', 'kB/s',
        ]);
    }
}
