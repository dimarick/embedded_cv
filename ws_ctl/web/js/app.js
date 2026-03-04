import Viewports from "./Viewports.js";

document.onreadystatechange = function() {
    window.viewports = new Viewports(document.getElementById("app-viewports"));
}