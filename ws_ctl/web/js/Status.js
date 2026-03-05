import Telemetry from "./Telemetry.js";

export default class Status {
    #dbName;
    #storeName;
    #db;
    #container;
    constructor() {
        document.addEventListener(Telemetry.EVENT_NAME_TELEMETRY, (event) => this.onTelemetry(event.detail));
    }

    onTelemetry(event) {
        if (event.command !== 'STATUS') {
            return;
        }

        const component = event.args.shift();
        while (event.args.length >= 2) {
            const property = event.args.shift();
            const value = event.args.shift();
            this.setStatus(component, property, value);
        }
    }

    setStatus(component, property, value) {
        switch (component) {
            case "calibration":
                this.setCalibrationProperty(property, value);
                break;
            case "camera":
                this.setCameraProperty(property, value);
                break;
        }
    }

    setCalibrationProperty(property, value) {

    }

    setCameraProperty(property, value) {
        switch (property) {
            case "resolution.w":
        }
    }
}