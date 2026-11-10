// Search as you type in the header field (see _templates/page.html), using the
// Pagefind index that docs/conf.py builds into pagefind/ after each HTML build.
// Results are grouped by page; each entry is the section that matched. When
// the index cannot be loaded (no index, or a page opened from file://), the
// form keeps its normal action and opens Sphinx's search page.
(function () {
  var form = document.querySelector(".site-header-search");
  if (!form) return;
  var header = document.querySelector(".site-header");
  var input = form.querySelector("input");
  var box = form.querySelector(".search-results");
  var root = new URL(document.documentElement.dataset.content_root || "./", location.href);
  var MAX_PAGES = 6;
  var MAX_SECTIONS = 3;

  var loading = null;
  var pagefind = null;
  var options = [];
  var active = -1;
  var query = 0;
  var timer = null;

  function load() {
    if (!loading) {
      loading = import(new URL("pagefind/pagefind.js", root).href)
        .then(function (pf) {
          // result URLs relative to the docs root, wherever that is hosted
          return pf.options({ baseUrl: root.pathname, excerptLength: 16 }).then(function () {
            pagefind = pf;
            return pf;
          });
        })
        .catch(function () {
          pagefind = null;
          return null;
        });
    }
    return loading;
  }

  function open() {
    box.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }

  function close() {
    box.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    active = -1;
  }

  function setActive(i) {
    if (active >= 0 && options[active]) options[active].classList.remove("is-active");
    active = i;
    if (active < 0 || !options[active]) {
      input.removeAttribute("aria-activedescendant");
      return;
    }
    options[active].classList.add("is-active");
    input.setAttribute("aria-activedescendant", options[active].id);
    options[active].scrollIntoView({ block: "nearest" });
  }

  function option(url, title, excerpt) {
    var a = document.createElement("a");
    a.className = "search-result";
    a.href = url;
    a.id = "search-result-" + options.length;
    a.setAttribute("role", "option");
    var t = document.createElement("span");
    t.className = "search-result-title";
    t.textContent = title;
    a.appendChild(t);
    if (excerpt) {
      var e = document.createElement("span");
      e.className = "search-result-excerpt";
      // Pagefind's excerpt is escaped page text with <mark> around the matches
      e.innerHTML = excerpt;
      a.appendChild(e);
    }
    options.push(a);
    return a;
  }

  function render(pages, q) {
    box.textContent = "";
    options = [];
    active = -1;
    if (!pages.length) {
      var empty = document.createElement("p");
      empty.className = "search-empty";
      empty.textContent = "No results for “" + q + "”.";
      box.appendChild(empty);
      return;
    }
    pages.forEach(function (page) {
      var group = document.createElement("div");
      group.className = "search-group";
      group.setAttribute("role", "group");
      var label = document.createElement("div");
      label.className = "search-group-title";
      label.textContent = page.meta.title;
      group.setAttribute("aria-label", page.meta.title);
      group.appendChild(label);
      page.entries.forEach(function (s) {
        group.appendChild(option(s.url, s.title, s.excerpt));
      });
      box.appendChild(group);
    });
  }

  // carlanomaly.datasets.RGBDataset.__getitem__ -> RGBDataset.__getitem__;
  // carlanomaly.download.require_parts -> download.require_parts
  function shortName(id) {
    var parts = id.split(".").slice(1);
    var cls = parts.findIndex(function (p) { return /^[A-Z]/.test(p); });
    return parts.slice(cls >= 0 ? cls : 0).join(".") || id;
  }

  // The places on a page to link to: Pagefind's sections (headings with an id)
  // and, on API pages, the documented objects (<dt id="carlanomaly...">), so a
  // search for a method lands on that method. Each match counts for the
  // innermost of these that contains it; entries whose name contains a search
  // term come first.
  function entries(page, q) {
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    var base = page.url.split("#")[0];
    var anchors = (page.anchors || []).filter(function (a) {
      return /^h[1-6]$/.test(a.element) || (a.element === "dt" && a.id.indexOf("carlanomaly.") === 0);
    });
    var objects = {};
    var taken = {};
    (page.locations || []).forEach(function (loc) {
      var inner = null;
      anchors.forEach(function (a) { if (a.location <= loc) inner = a; });
      if (!inner || inner.element !== "dt") return;
      taken[loc] = true;
      var o = objects[inner.id];
      if (!o) {
        o = objects[inner.id] = {
          url: base + "#" + inner.id,
          title: shortName(inner.id),
          excerpt: "",
          signature: inner.text.replace(/\[source\]$/, ""),
          hits: 0,
        };
      }
      o.hits += 1;
    });
    // objects named after a search term, even when Pagefind matched them
    // only through their indexed name
    anchors.forEach(function (a) {
      if (a.element !== "dt" || objects[a.id]) return;
      var name = shortName(a.id).toLowerCase();
      if (!terms.some(function (t) { return name.indexOf(t) >= 0; })) return;
      objects[a.id] = { url: base + "#" + a.id, title: shortName(a.id), excerpt: "",
                        signature: a.text.replace(/\[source\]$/, ""), hits: 0 };
    });
    var list = Object.keys(objects).map(function (id) {
      var o = objects[id];
      var span = document.createElement("span");
      span.textContent = o.signature.length > 110 ? o.signature.slice(0, 107) + "…" : o.signature;
      o.excerpt = span.innerHTML;
      return o;
    });
    (page.sub_results || []).forEach(function (s) {
      var hits = (s.locations || []).filter(function (loc) { return !taken[loc]; }).length;
      if (hits) list.push({ url: s.url, title: s.title, excerpt: s.excerpt, hits: hits });
    });
    if (!list.length) list.push({ url: page.url, title: page.meta.title, excerpt: page.excerpt, hits: 0 });
    // a name equal to a search term (PixelEvaluator) beats one containing it
    // (PixelEvaluator.update); then more matches, then the shorter name
    list.forEach(function (e) {
      var name = e.title.toLowerCase();
      var last = name.split(".").pop();
      // "pixel evaluator" names PixelEvaluator too
      var exact = terms.concat([terms.join("")]).some(function (t) { return name === t || last === t; });
      var contains = terms.some(function (t) { return name.indexOf(t) >= 0; });
      e.score = e.hits + (exact ? 200 : contains ? 100 : 0);
    });
    list.sort(function (a, b) { return b.score - a.score || a.title.length - b.title.length; });
    return list.slice(0, MAX_SECTIONS);
  }

  function search() {
    var q = input.value.trim();
    var mine = ++query;
    if (!q) {
      close();
      return;
    }
    load().then(function (pf) {
      if (!pf) return;
      return pf.search(q).then(function (res) {
        if (mine !== query) return;
        return Promise.all(res.results.slice(0, MAX_PAGES).map(function (r) { return r.data(); }));
      }).then(function (pages) {
        if (!pages || mine !== query) return;
        // pages whose section or API name contains the search term first
        pages.forEach(function (page, i) {
          page.entries = entries(page, q);
          page.rank = (page.entries[0].score >= 100 ? 0 : 1) * 1000 + i;
        });
        pages.sort(function (a, b) { return a.rank - b.rank; });
        render(pages, q);
        open();
      });
    });
  }

  function closeOverlay() {
    if (header) header.classList.remove("search-open");
  }

  input.addEventListener("focus", function () {
    load();
    if (input.value.trim() && options.length) open();
  });

  input.addEventListener("input", function () {
    clearTimeout(timer);
    timer = setTimeout(search, 120);
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (box.hidden || !options.length) return;
      e.preventDefault();
      var step = e.key === "ArrowDown" ? 1 : -1;
      setActive((active + step + options.length) % options.length);
    } else if (e.key === "Escape") {
      if (!box.hidden) {
        close();
      } else {
        input.blur();
        closeOverlay();
      }
    }
  });

  // Enter opens the selected result, or the first one; without the index the
  // form submits to Sphinx's search page.
  form.addEventListener("submit", function (e) {
    if (!pagefind || !options.length) return;
    e.preventDefault();
    location.href = options[Math.max(active, 0)].href;
  });

  box.addEventListener("mousemove", function (e) {
    var a = e.target.closest(".search-result");
    if (a) setActive(options.indexOf(a));
  });

  document.addEventListener("mousedown", function (e) {
    if (!form.contains(e.target)) {
      close();
      if (!e.target.closest(".site-header-search-button")) closeOverlay();
    }
  });
})();
