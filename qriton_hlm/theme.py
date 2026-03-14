"""
Qriton UI theme — shared across all HLM demos.
Dark sidebar, fixed header, Qriton color scheme.

Usage:
    from qriton_hlm.theme import qriton_theme, qriton_css, QRITON_JS
    with gr.Blocks(theme=qriton_theme(), css=qriton_css("App Name"), js=QRITON_JS) as app:
        ...
"""
import gradio as gr


def qriton_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e6f2f7", c100="#b3d9e8", c200="#80c0d9",
            c300="#4da7ca", c400="#2694bb", c500="#0077a8",
            c600="#006a97", c700="#005a80", c800="#004a69",
            c900="#003a52", c950="#002a3b",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f6f6f4", c100="#e0e0e0", c200="#cccccc",
            c300="#999999", c400="#777777", c500="#666666",
            c600="#555555", c700="#444444", c800="#333333",
            c900="#1a1a1a", c950="#0c0c0c",
        ),
        neutral_hue=gr.themes.Color(
            c50="#faf9f7", c100="#f6f6f4", c200="#e0e0e0",
            c300="#cccccc", c400="#999999", c500="#666666",
            c600="#555555", c700="#444444", c800="#1a1a1a",
            c900="#0c0c0c", c950="#050505",
        ),
        font=["Inter", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "Consolas", "monospace"],
    ).set(
        body_background_fill="#faf9f7",
        body_text_color="#0c0c0c",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e0e0e0",
        block_shadow="none",
        button_primary_background_fill="#0c0c0c",
        button_primary_background_fill_hover="#1a1a1a",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#0c0c0c",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f6f6f4",
        button_secondary_text_color="#0c0c0c",
        button_secondary_border_color="#e0e0e0",
        input_background_fill="#ffffff",
        input_border_color="#e0e0e0",
        input_border_width="1px",
        slider_color="#0077a8",
        checkbox_label_text_color="#0c0c0c",
    )


def qriton_css(app_name="HLM", subtitle="Qriton Technologies"):
    """Generate CSS with app header and sidebar."""
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* -- Fixed header -- */
    #qriton-header {{
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 48px;
        background: #0c0c0c;
        color: #fff;
        display: flex;
        align-items: center;
        padding: 0 20px;
        z-index: 200;
        font-family: 'Inter', sans-serif;
        border-bottom: 1px solid #222;
    }}
    #qriton-header .header-title {{
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 18px;
        font-weight: 400;
        letter-spacing: -0.01em;
        color: #fff;
    }}
    #qriton-header .header-sub {{
        font-size: 11px;
        color: #666;
        margin-left: 12px;
        font-weight: 400;
    }}
    #qriton-header .header-accent {{
        color: #0077a8;
    }}
    #qriton-header .header-toggle {{
        margin-left: auto;
        background: none;
        border: 1px solid #333;
        color: #999;
        width: 30px; height: 30px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s;
    }}
    #qriton-header .header-toggle:hover {{
        color: #fff;
        border-color: #555;
    }}

    .gradio-container {{
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        padding-top: 48px !important;
    }}

    /* -- Sidebar -- */
    div.tab-wrapper,
    .tab-wrapper {{
        position: fixed !important;
        left: 0 !important;
        top: 48px !important;
        width: 200px !important;
        height: calc(100vh - 48px) !important;
        background: #0c0c0c !important;
        z-index: 100 !important;
        border-right: 1px solid #1a1a1a !important;
        padding: 0 !important;
        padding-bottom: 0 !important;
        transition: width 0.2s ease !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }}
    div.tab-wrapper div.tab-container,
    .tab-wrapper .tab-container {{
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        width: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overflow: visible auto !important;
        padding: 8px 0 !important;
        gap: 0 !important;
    }}
    div.tab-wrapper div.tab-container::after,
    .tab-wrapper .tab-container::after {{
        display: none !important;
    }}
    div.tab-wrapper div.tab-container button,
    .tab-wrapper .tab-container button {{
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        color: #999 !important;
        background: transparent !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        padding: 10px 16px !important;
        margin: 0 !important;
        text-align: left !important;
        white-space: nowrap !important;
        border-radius: 0 !important;
        transition: all 0.12s ease !important;
        display: block !important;
        width: 100% !important;
        overflow: hidden !important;
        line-height: 1.4 !important;
        height: auto !important;
        flex-shrink: 0 !important;
    }}
    div.tab-wrapper div.tab-container button:hover,
    .tab-wrapper .tab-container button:hover {{
        color: #eee !important;
        background: #1a1a1a !important;
    }}
    div.tab-wrapper div.tab-container button.selected,
    .tab-wrapper .tab-container button.selected {{
        color: #fff !important;
        font-weight: 500 !important;
        border-left-color: #0077a8 !important;
        background: rgba(0, 119, 168, 0.15) !important;
    }}
    div.tab-wrapper div.tab-container button.selected::after,
    .tab-wrapper .tab-container button.selected::after {{
        display: none !important;
    }}
    div.tab-wrapper .overflow-menu,
    .tab-wrapper .overflow-menu {{
        display: none !important;
    }}

    /* -- Content area -- */
    .tabs {{ position: relative !important; }}
    .tabitem {{
        margin-left: 200px !important;
        padding: 24px 32px !important;
        min-height: calc(100vh - 48px) !important;
        transition: margin-left 0.2s ease !important;
    }}

    /* -- Collapsed sidebar -- */
    body.sidebar-collapsed .tab-wrapper {{
        width: 0 !important;
        padding: 0 !important;
        border-right: none !important;
        overflow: hidden !important;
    }}
    body.sidebar-collapsed .tabitem {{
        margin-left: 0 !important;
    }}

    /* -- Buttons -- */
    .gr-button {{
        border-radius: 100px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 8px 24px !important;
        transition: all 0.12s ease !important;
    }}
    .gr-button-primary {{
        letter-spacing: 0.02em;
        box-shadow: none !important;
    }}

    /* -- Typography -- */
    h1, h2, h3 {{ color: #0c0c0c !important; }}
    h1 {{
        font-family: Georgia, 'Times New Roman', serif !important;
        font-weight: 400 !important;
        font-size: 1.8em !important;
        letter-spacing: -0.02em !important;
        margin-top: 0 !important;
    }}
    h3 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #333 !important;
    }}

    /* -- Responsive -- */
    @media (max-width: 900px) {{
        .tab-wrapper {{ width: 0 !important; overflow: hidden !important; }}
        .tabitem {{ margin-left: 0 !important; padding: 16px !important; }}
        body.sidebar-open .tab-wrapper {{ width: 200px !important; overflow: visible !important; }}
    }}
    @media (max-width: 600px) {{
        .tabitem {{ padding: 10px !important; }}
        h1 {{ font-size: 1.3em !important; }}
        #qriton-header .header-sub {{ display: none; }}
    }}

    /* -- Scrollbars -- */
    .tab-wrapper .tab-container::-webkit-scrollbar {{ width: 3px; }}
    .tab-wrapper .tab-container::-webkit-scrollbar-track {{ background: transparent; }}
    .tab-wrapper .tab-container::-webkit-scrollbar-thumb {{ background: #333; border-radius: 3px; }}
    """


QRITON_JS = """
() => {
    function setupSidebar() {
        if (document.getElementById('qriton-header')) return;

        const title = document.title || 'HLM';
        const parts = title.split('--').map(s => s.trim());
        const appName = parts[0] || 'HLM';
        const subtitle = parts.length > 1 ? parts[1] : 'Qriton Technologies';

        const header = document.createElement('div');
        header.id = 'qriton-header';
        header.innerHTML = `
            <span class="header-title">${appName}</span>
            <span class="header-sub">${subtitle}</span>
            <button class="header-toggle" id="sidebar-btn" title="Toggle menu">&#9776;</button>
        `;
        document.body.prepend(header);

        document.getElementById('sidebar-btn').onclick = () => {
            document.body.classList.toggle('sidebar-collapsed');
            if (window.innerWidth <= 900) {
                document.body.classList.toggle('sidebar-open');
            }
        };
    }

    function forceSidebarLayout() {
        const wrapper = document.querySelector('.tab-wrapper');
        if (!wrapper) return;
        wrapper.style.setProperty('position', 'fixed', 'important');
        wrapper.style.setProperty('left', '0', 'important');
        wrapper.style.setProperty('top', '48px', 'important');
        wrapper.style.setProperty('width', '200px', 'important');
        wrapper.style.setProperty('height', 'calc(100vh - 48px)', 'important');
        wrapper.style.setProperty('background', '#0c0c0c', 'important');
        wrapper.style.setProperty('z-index', '100', 'important');
        wrapper.style.setProperty('flex-direction', 'column', 'important');
        wrapper.style.setProperty('padding-bottom', '0', 'important');

        const container = wrapper.querySelector('.tab-container');
        if (container) {
            container.style.setProperty('display', 'flex', 'important');
            container.style.setProperty('flex-direction', 'column', 'important');
            container.style.setProperty('height', '100%', 'important');
            container.style.setProperty('overflow-y', 'auto', 'important');
            container.style.setProperty('overflow-x', 'hidden', 'important');
            container.querySelectorAll('button').forEach(btn => {
                btn.style.setProperty('display', 'block', 'important');
                btn.style.setProperty('width', '100%', 'important');
                btn.style.setProperty('height', 'auto', 'important');
                btn.style.setProperty('text-align', 'left', 'important');
                btn.style.setProperty('flex-shrink', '0', 'important');
            });
        }

        const overflow = wrapper.querySelector('.overflow-menu');
        if (overflow) overflow.style.setProperty('display', 'none', 'important');
    }

    setTimeout(setupSidebar, 300);
    setTimeout(forceSidebarLayout, 500);
    setTimeout(forceSidebarLayout, 1000);
    setTimeout(forceSidebarLayout, 2000);
}
"""
