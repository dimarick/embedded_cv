import Viewports from "./Viewports";

document.onreadystatechange = function() {
    window.viewports = new Viewports(document.getElementById("app-viewports"));
}