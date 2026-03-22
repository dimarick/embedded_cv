import Socket from "./Socket.js";

export default class Telemetry {
    static EVENT_NAME_TELEMETRY = 'socket.tm.received'
    #cvSocket
    #calibSocket

    constructor() {
        this.#cvSocket = new Socket('/cv_tm', (data) => this.onMessage('cv', data))
        this.#calibSocket = new Socket('/cv_calib_tm', (data) => this.onMessage('calib', data))
        this.#cvSocket.connect();
        this.#calibSocket.connect();
    }

    async onMessage(subsystem, data) {
        const text = data instanceof Blob ? await data.text() : data;
        const parsedData = JSON.parse(text)
        const command = parsedData.shift();
        const args = parsedData;
        document.dispatchEvent(new CustomEvent(Telemetry.EVENT_NAME_TELEMETRY, {detail: {command, args, subsystem}}));
    }
}