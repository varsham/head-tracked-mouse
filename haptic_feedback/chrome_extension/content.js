function isButtonLike(el) {
	if (!el || !el.tagName) return false;

	const tag = el.tagName;
	if (tag === "BUTTON" || tag === "A") return true;

	const role = el.getAttribute && el.getAttribute("role");
	if (role === "button" || role === "link") return true;

	if (el.type === "submit" || el.type === "button") return true;

	return window.getComputedStyle(el).cursor === "pointer";
}

function findButtonAncestor(el) {
	let node = el;
	let depth = 0;
	while (node && depth < 5) {
		if (isButtonLike(node)) return node;
		node = node.parentElement;
		depth++;
	}
	return null;
}

let lastState = null;

function reportHover(hovering) {
	if (hovering === lastState) return;
	lastState = hovering;
	chrome.runtime.sendMessage({ type: "hover", hovering });
}

document.addEventListener(
	"mouseover",
	(e) => reportHover(!!findButtonAncestor(e.target)),
	true
);

document.addEventListener(
	"mouseout",
	(e) => {
		if (!e.relatedTarget) reportHover(false);
	},
	true
);
