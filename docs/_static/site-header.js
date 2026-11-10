// Site header behaviour (see _templates/page.html), loaded deferred on every page.
// Adapted from the pytorch-ood docs, without the Read the Docs search hooks.
// The results under the search field come from search.js.

var siteHeader = document.querySelector(".site-header");
var headerInput = document.querySelector(".site-header-search input");

// Below Furo's breakpoint the field is hidden; it opens over the header.
function openHeaderSearch() {
  if (headerInput.offsetParent === null) siteHeader.classList.add("search-open");
  headerInput.focus();
}

// Ctrl/Cmd+K or "/" focuses the header search, as on most doc sites.
document.addEventListener("keydown", function (e) {
  var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  if (!(((e.ctrlKey || e.metaKey) && e.key === "k") || (e.key === "/" && !typing))) return;
  if (!headerInput) return;
  e.preventDefault();
  openHeaderSearch();
});

document.querySelectorAll(".site-header-search-button").forEach(function (button) {
  button.addEventListener("click", openHeaderSearch);
});
