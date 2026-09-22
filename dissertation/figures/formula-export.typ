#set page(width: auto, height: auto, margin: 18pt, fill: white)
#set text(font: "New Computer Modern", size: 22pt)
#show math.equation: set text(font: "New Computer Modern Math")

#context {
  let which = sys.inputs.at("formula", default: "mi")
  if which == "mi" {
    $
      I(Theta; cal(M)) = sum_(theta in Theta) sum_(m in cal(M))
      p(theta, m) log frac(p(theta, m), p(theta) p(m))
    $
  } else if which == "sc" {
    $
      "SC" = frac(1, |Theta|) sum_(theta in Theta) max_(m in cal(M)) p(m | theta)
    $
  } else if which == "cic" {
    $
      "CIC" = D_"KL" ( pi^("drive")(dot | o, m) parallel bb(E)_(m' tilde p(m)) [pi^("drive")(dot | o, m')] )
    $
  } else if which == "cde" {
    $
      "CDE" = bb(E)_(s, thin m' tilde "Unif"(cal(M))) thin
      norm(mu(o_s, m') - mu(o_s, m))_2
    $
  }
}
