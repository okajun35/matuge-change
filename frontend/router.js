// ハッシュルーティング。DOM に触れない純粋な判定だけを持ち、node からもテストできるようにする。
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Router = api;
})(typeof self !== 'undefined' ? self : this, function () {
  const ROUTES = ['extract', 'video', 'catalog'];

  const PANELS = {
    extract: ['controls', 'status', 'stage'],
    video: ['videoMode'],
    catalog: ['catalogPanel'],
  };

  const ALL_PANELS = ['controls', 'status', 'stage', 'catalogPanel', 'videoMode'];

  function parseRoute(hash) {
    const name = String(hash || '').replace(/^#\/?/, '');
    return ROUTES.includes(name) ? name : 'extract';
  }

  function hashFor(route) {
    return `#/${parseRoute(`#/${route}`)}`;
  }

  function panelVisibility(route) {
    const shown = PANELS[parseRoute(`#/${route}`)];
    const visibility = {};
    for (const id of ALL_PANELS) visibility[id] = shown.includes(id);
    return visibility;
  }

  return { ROUTES, parseRoute, hashFor, panelVisibility };
});
