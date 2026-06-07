$(document).ready(function () {
  // add toggle functionality to abstract, award and bibtex buttons
  $("a.abstract").click(function () {
    $(this).parent().parent().find(".abstract.hidden").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.award").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.bibtex").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden").toggleClass("open");
  });
  $("a").removeClass("waves-effect waves-light");

  // bootstrap-toc
  if ($("#toc-sidebar").length) {
    // remove related publications years from the TOC
    $(".publications h2").each(function () {
      $(this).attr("data-toc-skip", "");
    });
    var navSelector = "#toc-sidebar";
    var $myNav = $(navSelector);
    Toc.init($myNav);
    $("body").scrollspy({
      target: navSelector,
    });
  }

  // add css to jupyter notebooks
  const cssLink = document.createElement("link");
  cssLink.href = "../css/jupyter.css";
  cssLink.rel = "stylesheet";
  cssLink.type = "text/css";

  let jupyterTheme = determineComputedTheme();

  $(".jupyter-notebook-iframe-container iframe").each(function () {
    $(this).contents().find("head").append(cssLink);

    if (jupyterTheme == "dark") {
      $(this).bind("load", function () {
        $(this).contents().find("body").attr({
          "data-jp-theme-light": "false",
          "data-jp-theme-name": "JupyterLab Dark",
        });
      });
    }
  });

  // trigger popovers
  $('[data-toggle="popover"]').popover({
    trigger: "hover",
  });

  // add shadow to navbar once the page is scrolled
  const navbar = document.getElementById("navbar");
  if (navbar) {
    const onScroll = function () {
      navbar.classList.toggle("scrolled", window.scrollY > 8);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  // propagate each publication's venue color to its entry as an accent
  document.querySelectorAll("ol.bibliography > li").forEach(function (li) {
    const badge = li.querySelector(".abbr abbr");
    if (!badge) return;
    const color = badge.style.backgroundColor;
    if (color) {
      li.style.setProperty("--venue-color", color);
    }
  });

  // scroll-reveal animation for sections, cards and publications
  const revealTargets = document.querySelectorAll(
    ".home-section-title, .news, .featured-posts .card, ol.bibliography > li, .projects .grid-item, .reveal-on-scroll"
  );
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (revealTargets.length && "IntersectionObserver" in window && !prefersReduced) {
    revealTargets.forEach(function (el, i) {
      el.classList.add("reveal-init");
      el.style.setProperty("--reveal-delay", (i % 6) * 60 + "ms");
    });
    const io = new IntersectionObserver(
      function (entries, obs) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("reveal-in");
            obs.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );
    revealTargets.forEach(function (el) {
      io.observe(el);
    });
  }

  // live research stats from Semantic Scholar with count-up animation
  const statsEl = document.querySelector(".research-stats");
  if (statsEl) {
    const id = statsEl.getAttribute("data-semantic-scholar-id");
    const valueEls = statsEl.querySelectorAll(".stat-value");
    const animateCount = function (el, target) {
      if (prefersReduced || target <= 0) {
        el.textContent = target.toLocaleString();
        return;
      }
      const duration = 1100;
      const start = performance.now();
      const step = function (now) {
        const p = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(target * eased).toLocaleString();
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    if (id) {
      fetch(
        "https://api.semanticscholar.org/graph/v1/author/" +
          encodeURIComponent(id) +
          "?fields=paperCount,citationCount,hIndex"
      )
        .then(function (r) {
          if (!r.ok) throw new Error("status " + r.status);
          return r.json();
        })
        .then(function (data) {
          valueEls.forEach(function (el) {
            const key = el.getAttribute("data-stat");
            const v = typeof data[key] === "number" ? data[key] : 0;
            el.setAttribute("data-count", v);
          });
          // animate now if already visible, else on reveal
          const trigger = function () {
            valueEls.forEach(function (el) {
              animateCount(el, parseInt(el.getAttribute("data-count"), 10) || 0);
            });
          };
          if ("IntersectionObserver" in window && !prefersReduced) {
            const so = new IntersectionObserver(
              function (entries, obs) {
                entries.forEach(function (e) {
                  if (e.isIntersecting) {
                    trigger();
                    obs.disconnect();
                  }
                });
              },
              { threshold: 0.3 }
            );
            so.observe(statsEl);
          } else {
            trigger();
          }
        })
        .catch(function () {
          statsEl.classList.add("stats-error");
          const src = statsEl.querySelector(".stat-source");
          if (src) src.textContent = "stats unavailable";
          valueEls.forEach(function (el) {
            el.textContent = "—";
          });
        });
    }
  }

  // live-enhance native GitHub repo cards from the public API
  const langColors = {
    Python: "#3572A5",
    JavaScript: "#f1e05a",
    TypeScript: "#3178c6",
    "Jupyter Notebook": "#DA5B0B",
    "C++": "#f34b7d",
    C: "#555555",
    Cuda: "#3A4E3A",
    Shell: "#89e051",
    HTML: "#e34c26",
    CSS: "#563d7c",
    Rust: "#dea584",
    Go: "#00ADD8",
    Java: "#b07219",
  };
  const fmt = function (n) {
    if (typeof n !== "number") return n;
    return n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : n.toLocaleString();
  };

  document.querySelectorAll(".repo-card[data-repo]").forEach(function (card) {
    const slug = card.getAttribute("data-repo");
    card.classList.add("repo-loading");
    fetch("https://api.github.com/repos/" + slug, {
      headers: { Accept: "application/vnd.github+json" },
    })
      .then(function (r) {
        if (!r.ok) throw new Error("status " + r.status);
        return r.json();
      })
      .then(function (d) {
        const starsEl = card.querySelector("[data-repo-stars]");
        const forksEl = card.querySelector("[data-repo-forks]");
        const descEl = card.querySelector("[data-repo-desc]");
        if (starsEl && typeof d.stargazers_count === "number") starsEl.textContent = fmt(d.stargazers_count);
        if (forksEl && typeof d.forks_count === "number") forksEl.textContent = fmt(d.forks_count);
        if (descEl && d.description && !descEl.textContent.trim()) descEl.textContent = d.description;

        const langWrap = card.querySelector("[data-repo-lang]");
        if (langWrap && d.language) {
          langWrap.hidden = false;
          const nameSpan = langWrap.querySelector("[data-repo-lang-name]");
          if (nameSpan) nameSpan.textContent = d.language;
          const dot = langWrap.querySelector("[data-repo-lang-dot]");
          if (dot && langColors[d.language]) dot.style.background = langColors[d.language];
        }
      })
      .catch(function () {
        /* keep baked fallback values */
      })
      .finally(function () {
        card.classList.remove("repo-loading");
      });
  });

  // live-enhance native GitHub profile cards
  document.querySelectorAll(".repo-user-card[data-user]").forEach(function (card) {
    const login = card.getAttribute("data-user");
    fetch("https://api.github.com/users/" + login, {
      headers: { Accept: "application/vnd.github+json" },
    })
      .then(function (r) {
        if (!r.ok) throw new Error("status " + r.status);
        return r.json();
      })
      .then(function (d) {
        const set = function (sel, v) {
          const el = card.querySelector(sel);
          if (el && (typeof v === "number" || v)) el.textContent = typeof v === "number" ? v.toLocaleString() : v;
        };
        if (d.name) set("[data-user-name]", d.name);
        set("[data-user-repos]", d.public_repos);
        set("[data-user-followers]", d.followers);
        set("[data-user-following]", d.following);
      })
      .catch(function () {
        /* keep baked fallback values */
      });
  });

  // spotlight cursor glow on cards and publication entries
  if (window.matchMedia("(pointer: fine)").matches && !prefersReduced) {
    const spotlightEls = document.querySelectorAll(
      ".card, ol.bibliography > li, .research-stats .stat, .featured-posts .card, .repo-card, .repo-user-card"
    );
    spotlightEls.forEach(function (el) {
      el.classList.add("has-spotlight");
      el.addEventListener("pointermove", function (e) {
        const r = el.getBoundingClientRect();
        el.style.setProperty("--mx", e.clientX - r.left + "px");
        el.style.setProperty("--my", e.clientY - r.top + "px");
      });
    });
  }

  // subtle pointer-tilt on the hero portrait
  const portrait = document.querySelector(".hero-portrait .portrait-frame");
  if (portrait && !prefersReduced && window.matchMedia("(pointer: fine)").matches) {
    const wrap = portrait.parentElement;
    const maxTilt = 7;
    wrap.addEventListener("mousemove", function (e) {
      const r = portrait.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width - 0.5;
      const py = (e.clientY - r.top) / r.height - 0.5;
      portrait.style.transform =
        "perspective(800px) rotateY(" + px * maxTilt + "deg) rotateX(" + -py * maxTilt + "deg)";
    });
    wrap.addEventListener("mouseleave", function () {
      portrait.style.transform = "";
    });
  }
});
