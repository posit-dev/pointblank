/**
 * Reference Switcher Widget
 *
 * When only two reference sections exist (API + CLI), renders a segmented
 * button pair at the top of the sidebar.
 *
 * When three or more sections exist (API + CLI + MCP, etc.), renders a
 * dropdown select instead so the UI stays compact.
 *
 * Sections are detected from a data attribute on <body> injected at build
 * time: data-gd-ref-sections="api,cli,mcp"
 */

(function() {
    'use strict';

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initReferenceSwitcher);
    } else {
        initReferenceSwitcher();
    }

    // Section metadata: id → { label, icon (Bootstrap Icons class), pathSegment }
    var SECTIONS = {
        api: { label: 'Python API', icon: 'bi-code-square', path: '' },
        cli: { label: 'CLI', icon: 'bi-terminal', path: 'cli/' },
        mcp: { label: 'MCP Server', icon: 'bi-cpu', path: 'mcp/' }
    };

    function getEnabledSections() {
        // Read from body data attribute (set by build)
        var attr = document.body.getAttribute('data-gd-ref-sections');
        if (attr) {
            return attr.split(',').map(function(s) { return s.trim(); }).filter(Boolean);
        }
        // Fallback: detect from DOM
        var sections = ['api'];
        if (document.querySelector('a[href*="/reference/cli/"]') ||
            window.location.pathname.indexOf('/reference/cli/') !== -1) {
            sections.push('cli');
        }
        if (document.querySelector('a[href*="/reference/mcp/"]') ||
            window.location.pathname.indexOf('/reference/mcp/') !== -1) {
            sections.push('mcp');
        }
        return sections;
    }

    function getCurrentSection() {
        var path = window.location.pathname;
        if (path.indexOf('/reference/mcp/') !== -1) return 'mcp';
        if (path.indexOf('/reference/cli/') !== -1) return 'cli';
        if (path.indexOf('/reference/') !== -1) return 'api';
        return null;
    }

    function initReferenceSwitcher() {
        var current = getCurrentSection();
        if (!current) return; // Not on a reference page

        var sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        var enabled = getEnabledSections();
        if (enabled.length < 2) return; // No switcher needed

        var switcherContainer = document.createElement('div');
        switcherContainer.className = 'reference-switcher-container';

        if (enabled.length === 2) {
            // Segmented button (original behaviour)
            switcherContainer.innerHTML = buildSegmentedHtml(enabled, current);
            attachSegmentedHandlers(switcherContainer);
        } else {
            // Dropdown for 3+ sections
            switcherContainer.innerHTML = buildDropdownHtml(enabled, current);
            attachDropdownHandlers(switcherContainer);
        }

        // Insert at the top of the sidebar-menu-container
        var menuContainer = sidebar.querySelector('.sidebar-menu-container');
        if (!menuContainer) return;

        var filterContainer = menuContainer.querySelector('.sidebar-filter-container');
        if (filterContainer) {
            menuContainer.insertBefore(switcherContainer, filterContainer);
        } else {
            menuContainer.insertBefore(switcherContainer, menuContainer.firstChild);
        }
    }

    // ── Segmented button (2 sections) ──────────────────────────────────────

    function buildSegmentedHtml(sections, current) {
        var html = '<div class="reference-switcher">';
        for (var i = 0; i < sections.length; i++) {
            var id = sections[i];
            var meta = SECTIONS[id] || { label: id, icon: '', path: id + '/' };
            var active = id === current ? ' active' : '';
            html += '<button class="reference-switcher-btn' + active + '"'
                  + ' data-ref="' + id + '"'
                  + ' title="' + meta.label + '">'
                  + '<i class="bi ' + meta.icon + '"></i>'
                  + '<span>' + meta.label + '</span>'
                  + '</button>';
        }
        html += '</div>';
        return html;
    }

    function attachSegmentedHandlers(container) {
        var buttons = container.querySelectorAll('.reference-switcher-btn');
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].addEventListener('click', function() {
                navigateToReference(this.getAttribute('data-ref'));
            });
        }
    }

    // ── Dropdown (3+ sections) ─────────────────────────────────────────────

    function buildDropdownHtml(sections, current) {
        var currentMeta = SECTIONS[current] || { label: current, icon: '', path: current + '/' };
        var html = '<div class="reference-switcher-dropdown">';
        html += '<button class="reference-switcher-dropdown-toggle" aria-expanded="false" aria-haspopup="listbox">';
        html += '<i class="bi ' + currentMeta.icon + '"></i>';
        html += '<span class="reference-switcher-dropdown-label">' + currentMeta.label + '</span>';
        html += '<svg class="reference-switcher-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>';
        html += '</button>';
        html += '<ul class="reference-switcher-dropdown-menu" role="listbox">';
        for (var i = 0; i < sections.length; i++) {
            var id = sections[i];
            var meta = SECTIONS[id] || { label: id, icon: '', path: id + '/' };
            var selected = id === current ? ' aria-selected="true"' : '';
            var activeClass = id === current ? ' active' : '';
            html += '<li class="reference-switcher-dropdown-item' + activeClass + '" role="option" data-ref="' + id + '"' + selected + '>';
            html += '<i class="bi ' + meta.icon + '"></i>';
            html += '<span>' + meta.label + '</span>';
            html += '</li>';
        }
        html += '</ul>';
        html += '</div>';
        return html;
    }

    function attachDropdownHandlers(container) {
        var toggle = container.querySelector('.reference-switcher-dropdown-toggle');
        var menu = container.querySelector('.reference-switcher-dropdown-menu');
        if (!toggle || !menu) return;

        toggle.addEventListener('click', function(e) {
            e.stopPropagation();
            var expanded = toggle.getAttribute('aria-expanded') === 'true';
            toggle.setAttribute('aria-expanded', String(!expanded));
            menu.classList.toggle('show');
        });

        var items = menu.querySelectorAll('.reference-switcher-dropdown-item');
        for (var i = 0; i < items.length; i++) {
            items[i].addEventListener('click', function() {
                navigateToReference(this.getAttribute('data-ref'));
            });
        }

        // Close on outside click
        document.addEventListener('click', function() {
            toggle.setAttribute('aria-expanded', 'false');
            menu.classList.remove('show');
        });

        // Close on Escape
        container.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                toggle.setAttribute('aria-expanded', 'false');
                menu.classList.remove('show');
                toggle.focus();
            }
        });
    }

    // ── Navigation ─────────────────────────────────────────────────────────

    function navigateToReference(refType) {
        var basePath = window.location.pathname.split('/reference/')[0];
        var meta = SECTIONS[refType] || { path: refType + '/' };
        window.location.href = basePath + '/reference/' + meta.path + 'index.html';
    }
})();
