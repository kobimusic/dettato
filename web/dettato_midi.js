import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const downloadLists = new WeakMap();

app.registerExtension({
    name: "Dettato.MidiDownloads",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DettatoSaveMIDI") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            onExecuted?.apply(this, arguments);
            if (!output?.midi?.length) return;

            let list = downloadLists.get(this);
            if (!list) {
                list = document.createElement("div");
                Object.assign(list.style, {
                    display: "grid",
                    alignContent: "start",
                    gap: "4px",
                    overflow: "auto",
                    fontSize: "12px",
                });
                downloadLists.set(this, list);
                const height = () => Math.min(24 * list.childElementCount + 4, 148);
                this.addDOMWidget("midi_downloads", "midi_downloads", list, {
                    serialize: false,
                    getMinHeight: height,
                    getMaxHeight: height,
                    getHeight: height,
                });
            }

            const links = output.midi.map(({ filename, subfolder, type }) => {
                const link = document.createElement("a");
                const params = new URLSearchParams({ filename, subfolder, type });
                link.href = api.apiURL(`/view?${params}`);
                link.download = filename;
                link.textContent = `Download ${filename}`;
                link.style.color = "var(--input-text, #ddd)";
                return link;
            });
            list.replaceChildren(...links);
            this.setSize([this.size[0], Math.max(this.size[1], this.computeSize()[1])]);
            this.setDirtyCanvas(true, true);
        };
    },
});
