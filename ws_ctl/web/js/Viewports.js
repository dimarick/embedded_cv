import Viewport from "./Viewport";

export default class Viewports {
    #viewportsElement;
    #viewports = [];


    constructor(viewportsElement) {
        this.#viewportsElement = viewportsElement;

        for (const element in viewportsElement.getElementsByClassName("viewport")) {
            this.#viewports = new Viewport(element);
        }
    }
}