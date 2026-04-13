/**
 * Video embedding enhancements for Great Docs
 *
 * Improves performance and UX for embedded videos:
 *  - YouTube: lightweight thumbnail placeholder with click-to-play
 *    (avoids loading heavy YouTube iframe until the user requests it)
 *  - Vimeo / other service iframes: IntersectionObserver lazy loading
 *  - Local <video> elements: sets preload="metadata" for faster page loads
 */
(function () {
  "use strict";

  // YouTube play-button SVG (same style as YouTube's own overlay)
  var PLAY_SVG =
    '<svg viewBox="0 0 68 48" width="68" height="48" xmlns="http://www.w3.org/2000/svg">' +
    '<path class="gd-play-bg" d="M66.52 7.74c-.78-2.93-2.49-5.41-5.42-6.19C55.79.13 34 0 34 0S12.21.13 6.9 1.55' +
    "C3.97 2.33 2.27 4.81 1.48 7.74.06 13.05 0 24 0 24s.06 10.95 1.48 16.26c.78 2.93 2.49 5.41 5.42 6.19" +
    'C12.21 47.87 34 48 34 48s21.79-.13 27.1-1.55c2.93-.78 4.64-3.26 5.42-6.19C67.94 34.95 68 24 68 24' +
    's-.06-10.95-1.48-16.26z" fill="#212121" fill-opacity="0.8"/>' +
    '<path d="M45 24 27 14v20z" fill="#fff"/>' +
    "</svg>";

  /**
   * Extract a YouTube video ID from an embed URL.
   */
  function getYouTubeId(src) {
    var m = src.match(
      /(?:youtube\.com|youtube-nocookie\.com)\/embed\/([^?&#/]+)/
    );
    return m ? m[1] : null;
  }

  /**
   * Replace YouTube iframes with a static thumbnail + play button.
   * The real iframe is loaded only when the user clicks or presses Enter/Space.
   */
  function enhanceYouTube() {
    var iframes = document.querySelectorAll(
      '.quarto-video iframe[src*="youtube.com/embed"],' +
        '.quarto-video iframe[src*="youtube-nocookie.com/embed"]'
    );

    for (var i = 0; i < iframes.length; i++) {
      (function (iframe) {
        var videoId = getYouTubeId(iframe.src);
        if (!videoId) return;

        var wrapper = iframe.closest(".quarto-video");
        if (!wrapper) return;

        var originalSrc = iframe.src;
        var title =
          iframe.getAttribute("title") ||
          iframe.getAttribute("aria-label") ||
          "Video";

        // Build placeholder
        var ph = document.createElement("div");
        ph.className = "gd-video-placeholder";
        ph.setAttribute("role", "button");
        ph.setAttribute("aria-label", "Play video: " + title);
        ph.setAttribute("tabindex", "0");

        // Thumbnail image (try maxresdefault, fall back to hqdefault)
        var thumb = document.createElement("img");
        thumb.className = "gd-video-thumb";
        thumb.alt = title;
        thumb.loading = "lazy";
        thumb.src =
          "https://img.youtube.com/vi/" + videoId + "/maxresdefault.jpg";
        thumb.onerror = function () {
          this.src =
            "https://img.youtube.com/vi/" + videoId + "/hqdefault.jpg";
          this.onerror = null;
        };

        // Play button overlay
        var play = document.createElement("div");
        play.className = "gd-video-play";
        play.innerHTML = PLAY_SVG;

        ph.appendChild(thumb);
        ph.appendChild(play);

        // Swap: hide iframe (and prevent it from loading), show placeholder
        iframe.removeAttribute("src");
        iframe.style.display = "none";
        wrapper.appendChild(ph);

        function load() {
          ph.remove();
          iframe.style.display = "";
          // Append autoplay so the video starts immediately after click
          var sep = originalSrc.indexOf("?") === -1 ? "?" : "&";
          iframe.src = originalSrc + sep + "autoplay=1";
        }

        ph.addEventListener("click", load);
        ph.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            load();
          }
        });
      })(iframes[i]);
    }
  }

  /**
   * Lazy-load non-YouTube iframes (Vimeo, Loom, etc.) via IntersectionObserver.
   * The src is deferred until the iframe scrolls near the viewport.
   */
  function lazyLoadIframes() {
    if (!("IntersectionObserver" in window)) return;

    var iframes = document.querySelectorAll(
      '.quarto-video iframe:not([src*="youtube.com"]):not([src*="youtube-nocookie.com"])'
    );

    var observer = new IntersectionObserver(
      function (entries) {
        for (var j = 0; j < entries.length; j++) {
          if (entries[j].isIntersecting) {
            var el = entries[j].target;
            if (el.dataset.gdSrc) {
              el.src = el.dataset.gdSrc;
              delete el.dataset.gdSrc;
            }
            observer.unobserve(el);
          }
        }
      },
      { rootMargin: "300px" }
    );

    for (var i = 0; i < iframes.length; i++) {
      var iframe = iframes[i];
      if (!iframe.src) continue;
      iframe.dataset.gdSrc = iframe.src;
      iframe.removeAttribute("src");
      observer.observe(iframe);
    }
  }

  /**
   * Set preload="metadata" on <video> elements that have no explicit preload,
   * so browsers download only enough to show the first frame and duration.
   */
  function enhanceVideoElements() {
    var videos = document.querySelectorAll("video");
    for (var i = 0; i < videos.length; i++) {
      if (!videos[i].hasAttribute("preload")) {
        videos[i].setAttribute("preload", "metadata");
      }
    }
  }

  // --- Entry point ---
  function init() {
    enhanceYouTube();
    lazyLoadIframes();
    enhanceVideoElements();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
