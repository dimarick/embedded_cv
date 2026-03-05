import Socket from "./Socket.js";

export default class Telemetry {
    static EVENT_NAME_TELEMETRY = 'socket.tm.received'
    #socket

    constructor() {
        this.#socket = new Socket('/cv_tm', (data) => this.onMessage(data))
        this.#socket.connect();
    }

    onMessage(data) {
        const parsedData = JSON.parse(data)
        const command = parsedData.shift();
        const args = parsedData;
        document.dispatchEvent(new CustomEvent(Telemetry.EVENT_NAME_TELEMETRY, {detail: {command, args}}));
    }
}