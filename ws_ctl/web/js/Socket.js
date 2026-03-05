export default class Socket {
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
        }, 2000)
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
    }
}
