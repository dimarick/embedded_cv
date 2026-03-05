import Viewports from "./Viewports.js";
import Telemetry from "./Telemetry.js";
import Logger from "./Logger.js";

document.onreadystatechange = function() {
    window.logger = new Logger(document.getElementById("logs-container"));
    window.telemetry = new Telemetry();
    window.viewports = new Viewports(document.getElementById("app-viewports"));
}