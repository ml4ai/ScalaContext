import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "org.ml4ai",
      scalaVersion := "2.12.6",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "ScalaContext",
    libraryDependencies ++= Seq(
      scalaTest % Test,
      "com.typesafe" % "config" % "1.3.2"
    )
  )

libraryDependencies ++= Seq(
  "com.github.haifengl" %% "smile-scala" % "1.5.1",
  "org.clulab" %% "processors-main" % "7.4.2",
)


publishTo := Some(Resolver.file("file", new File("/Users/shraddha/datascience/ScalaContext")))
