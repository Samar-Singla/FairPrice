export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const origin = new URL(env.DEMO_ORIGIN);
    url.protocol = origin.protocol;
    url.hostname = origin.hostname;
    url.port = origin.port;
    try {
      const response = await fetch(new Request(url, request), {redirect: "manual"});
      if (response.status >= 500) return offline();
      const headers = new Headers(response.headers);
      headers.set("X-Robots-Tag", "noindex, nofollow");
      headers.set("Cache-Control", "no-store");
      return new Response(response.body, {status: response.status, headers});
    } catch { return offline(); }
  }
};
function offline() {
  return new Response(`<!doctype html><meta name="viewport" content="width=device-width"><title>FairPrice demo offline</title><style>body{font:18px Arial,sans-serif;max-width:700px;margin:15vh auto;padding:24px}h1{font-size:48px;letter-spacing:-2px}p{line-height:1.6}</style><h1>FairPrice</h1><h2>The demonstration is offline.</h2><p>This research demo runs on its owner's Mac. It will be available when the computer and its Cloudflare tunnel are online.</p><button onclick="location.reload()">Try again</button>`, {status:503,headers:{"Content-Type":"text/html;charset=UTF-8","Cache-Control":"no-store","Retry-After":"60"}});
}
