const BRIDGE_URL = "http://127.0.0.1:5005";

chrome.runtime.onMessage.addListener((message) => {
	if (message.type !== "hover") return;

	fetch(BRIDGE_URL, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ hovering: message.hovering }),
	}).catch(() => {});
});
