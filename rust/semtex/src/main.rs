fn main() {
    
    // perform a search for "Barack Obama" and poll the result
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(search("Barry Obama"));
}

async fn search(mention: &str) {
    let client = reqwest::Client::new();

    let response = client.get("https://www.wikidata.org/w/api.php")
        .query(&[("action", "wbsearchentities"), ("format", "json"), ("language", "en"), ("search", mention)])
        .send()
        .await
        .unwrap();

    let body = response.text().await.unwrap();

    println!("{}", body);
}