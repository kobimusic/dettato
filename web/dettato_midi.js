import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const downloadLists = new WeakMap();
const rowHeight = 28;
const rowGap = 4;
const widgetMargin = 4;

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
                    gridTemplateColumns: "minmax(0, 1fr)",
                    gridAutoRows: `${rowHeight}px`,
                    gap: `${rowGap}px`,
                    overflowY: "auto",
                    boxSizing: "border-box",
                    padding: `${rowGap / 2}px 0`,
                    fontSize: "12px",
                });
                downloadLists.set(this, list);
                // ComfyUI subtracts both margins from the space available to the rows.
                const height = () => Math.min(list.childElementCount, 5) * (rowHeight + rowGap) + 2 * widgetMargin;
                this.addDOMWidget("midi_downloads", "midi_downloads", list, {
                    serialize: false,
                    margin: widgetMargin,
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
                link.title = link.textContent;
                Object.assign(link.style, {
                    lineHeight: `${rowHeight}px`,
                    padding: "0 6px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: "var(--input-text, #ddd)",
                });
                return link;
            });
            list.replaceChildren(...links);
            this.setSize([Math.max(this.size[0], 300), Math.max(this.size[1], this.computeSize()[1])]);
            this.setDirtyCanvas(true, true);
        };
    },
});
